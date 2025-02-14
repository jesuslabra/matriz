from flask import Flask, render_template, request, redirect, url_for, session, flash
import pandas as pd
import joblib
import mariadb
from flask_session import Session
from datetime import datetime

# Inicialización de la aplicación Flask
app = Flask(__name__)
app.secret_key = 'clave_secreta_para_sesiones'
app.config['SESSION_TYPE'] = 'filesystem'
Session(app)

# Configuración de la base de datos MySQL
db_config = {
    'host': 'localhost',
    'user': 'root',
    'password': '',
    'database': 'matriz_finalia'
}


# Cargar el modelo previamente guardado
regressor = joblib.load('modelo_crediticio.pkl')
X_train = joblib.load('X_train.pkl')

# Función para calcular el porcentaje de endeudamiento
def calcular_porcentaje_endeudamiento(monto_solicitado, ingreso_mensual, num_dependientes, meses_prestamo, tipo_vivienda, tipo_contrato, edad, renta_mensual=None, gasto_por_dependiente=200, tasa_interes=0.50):
    if ingreso_mensual <= 0 or meses_prestamo <= 0:
        raise ValueError("El ingreso mensual y los meses del préstamo deben ser mayores a 0.")

    # Calcular la mensualidad con interés
    if tasa_interes > 0:
        tasa_mensual = tasa_interes / 12
        mensualidad = monto_solicitado * (tasa_mensual * (1 + tasa_mensual) ** meses_prestamo) / ((1 + tasa_mensual) ** meses_prestamo - 1)
    else:
        mensualidad = monto_solicitado / meses_prestamo

    # Agregar renta si la vivienda es alquilada
    if tipo_vivienda == 'inquilino' and renta_mensual is not None:
        mensualidad += renta_mensual

    # Calcular gastos por dependientes
    gastos_dependientes = num_dependientes * gasto_por_dependiente

    # Calcular porcentaje de endeudamiento
    porcentaje_endeudamiento = ((mensualidad + gastos_dependientes) / ingreso_mensual) * 100

    # Ajustes por tipo de contrato
    if tipo_contrato in ['Asalariado', 'Jubilado']:
        porcentaje_endeudamiento *= 0.85
    elif tipo_contrato in ['Trabajador Independiente', 'Trabajador Eventual']:
        porcentaje_endeudamiento *= 1.1

    # Ajustes por edad
    if edad >= 45:
        porcentaje_endeudamiento *= 0.75
    elif edad < 25:
        porcentaje_endeudamiento *= 1.1

    # Ajustes por tipo de vivienda
    if tipo_vivienda == 'propia':
        porcentaje_endeudamiento *= 0.9

    return mensualidad, porcentaje_endeudamiento


# Funciones adicionales: calcular_capacidad_pago, clasificar_riesgo, calcular_score_final (mantener tus funciones previas).

# Función para clasificar el riesgo basado en el porcentaje
def clasificar_riesgo(score_final):
    if score_final < 700:
        return 'Riesgo Extremo'
    elif 700 <= score_final < 800:
        return 'Riesgo Alto'
    elif 800 <= score_final < 900:
        return 'Riesgo Tolerable'
    elif score_final >= 900:
        return 'Riesgo Aceptable'
    else:
        return 'Fuera de rango'



# Función corregida para calcular la capacidad de pago con validación
def calcular_capacidad_pago(ingreso_mensual, mensualidad, gasto_por_dependiente, num_dependientes, tipo_vivienda, renta_mensual=0):
    # Calcular los gastos por dependientes
    gastos_dependientes = num_dependientes * gasto_por_dependiente

    # Incluir renta mensual si la vivienda es alquilada
    if tipo_vivienda == 'inquilino' and renta_mensual is not None:
        gastos_dependientes += renta_mensual
    
    # Calcular la capacidad de pago
    capacidad_pago = ingreso_mensual - mensualidad - gastos_dependientes

    # Validar si la capacidad de pago es menor al 30% del ingreso
    capacidad_pago_porcentaje = (capacidad_pago / ingreso_mensual) * 100
    if capacidad_pago_porcentaje < 30:
        return capacidad_pago, False  # Retorna False para indicar rechazo del crédito
    
    return capacidad_pago, True  # Si es aceptable, retorna True

# Ajustar riesgo por tiempo en empleo y residencia
def ajustar_por_estabilidad(tiempo_empleo, tiempo_residencia):
    ajuste_empleo = min(10, (tiempo_empleo - 6) / 12 * 10) if tiempo_empleo >= 6 else -10
    ajuste_residencia = min(10, tiempo_residencia / 12 * 10)
    return ajuste_empleo + ajuste_residencia
    
# Función corregida para calcular el score final
def calcular_score_final(porcentaje, categoria_riesgo, prediccion_incumplimiento, tiempo_empleo, tiempo_residencia, score_crediticio):
    # Score basado en el porcentaje de endeudamiento
    score_endeudamiento = max(0, 100 - porcentaje)

    # Score basado en la categoría de riesgo
    mapping_riesgo = {
        'aceptable': 90,
        'tolerable': 70,
        'alto': 50,
        'extremo': 20
    }
    score_riesgo = mapping_riesgo.get(categoria_riesgo, 0)

    # Penalización extra para score crediticio bajo
    if score_crediticio < 410:
        prediccion_incumplimiento = min(100, prediccion_incumplimiento * 1.3)  # Aumenta 30%
    elif score_crediticio < 480:
        prediccion_incumplimiento = min(100, prediccion_incumplimiento * 1.1)  # Aumenta 10%

    # Score del modelo de predicción
    score_modelo = max(0, 100 - (prediccion_incumplimiento ** 0.8))  # Suaviza más el impacto

    # Ajuste por estabilidad (empleo y residencia)
    estabilidad = ajustar_por_estabilidad(tiempo_empleo, tiempo_residencia)

    # Bonus adicional por score crediticio
    if score_crediticio < 400:
        bonus_score_crediticio = -50  # Penalización fuerte para scores muy bajos
    elif score_crediticio < 500:
        bonus_score_crediticio = -20  # Penalización leve
    else:
        bonus_score_crediticio = ((score_crediticio - 500) / 500) * 8  # Máximo de 8 puntos (antes 10)

    # Ponderaciones ajustadas para evitar crecimiento excesivo
    score_final = (
        0.30 * score_endeudamiento + 
        0.30 * score_riesgo + 
        0.50 * score_modelo +  # Ajuste menor en predicción
        0.80 * estabilidad + 
        0.07 * bonus_score_crediticio  # Reducimos su impacto
    )

    # Penalización más gradual si el score crediticio es menor a 700
    if score_crediticio < 400:
        score_final *= 0.3  # Se reduce en 70%
    elif score_crediticio < 450:
        score_final *= 0.55  # Se reduce en 45%
    elif score_crediticio < 500:
        score_final *= 0.65  # Se reduce en 35%
    elif score_crediticio < 550:
        score_final *= 0.75  # Se reduce en 25%
    elif score_crediticio < 600:
        score_final *= 0.85  # Se reduce en 15%
    elif score_crediticio < 650:
        score_final *= 0.90  # Se reduce en 10%
    elif score_crediticio < 700:
        score_final *= 0.95  # Se reduce en 5%

    # Mantener dentro del rango permitido (0-1000)
    score_final = max(0, min(score_final * 10, 1000))

    return score_final


#####################################
# Función para conectar a MariaDB
def get_db_connection():
    return mariadb.connect(**db_config)

@app.route('/', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        connection = get_db_connection()
        cursor = connection.cursor()
        query = "SELECT * FROM usuarios WHERE usuario = ? AND password = ?"
        cursor.execute(query, (username, password))
        user = cursor.fetchone()
        cursor.close()
        connection.close()

        if user:
            session['username'] = username  # MariaDB no usa diccionarios por defecto
            session['user_id'] = user[0]  # user[0] es id_usuario
            flash('Inicio de sesión exitoso', 'success')
            return redirect(url_for('formulario'))
        else:
            flash('Usuario o contraseña incorrectos', 'danger')
    return render_template('login.html')

@app.route('/formulario')
def formulario():
    if 'username' not in session:
        flash('Por favor, inicia sesión primero.', 'warning')
        return redirect(url_for('login'))
    return render_template('formulario.html')

@app.route('/calcular_credito', methods=['POST'])
def calcular_credito():
    if 'username' not in session:
        flash('Por favor, inicia sesión primero.', 'warning')
        return redirect(url_for('login'))  
    # Recibir los datos del formulario
    nombre = request.form['nombre']
    apellido_paterno = request.form['apellido_paterno']
    apellido_materno = request.form['apellido_materno']
    monto_solicitado = float(request.form['monto_solicitado'])
    ingreso_mensual = float(request.form['ingreso_mensual'])
    num_dependientes = int(request.form['num_dependientes'])
    meses_prestamo = int(request.form['meses_prestamo'])
    tipo_vivienda = request.form['tipo_vivienda']
    tipo_contrato = request.form['tipo_contrato']
    edad = int(request.form['edad'])
    renta_mensual = float(request.form.get('renta_mensual', 0))  # Valor por defecto 0 si no se proporciona
    gasto_por_dependiente = float(request.form['gasto_por_dependiente'])
    estado_civil = request.form['estado_civil']
    grado_estudios = request.form['grado_estudios']
    tipo_ocupacion = request.form['tipo_ocupacion']
    frecuencia_pago = request.form['frecuencia_pago']
    tiempo_empleo = int(request.form['tiempo_empleo'])
    tiempo_residencia = int(request.form['tiempo_residencia'])

    # Calcular la mensualidad y el porcentaje de endeudamiento
    mensualidad, porcentaje = calcular_porcentaje_endeudamiento(
        monto_solicitado=monto_solicitado,
        ingreso_mensual=ingreso_mensual,
        num_dependientes=num_dependientes,
        meses_prestamo=meses_prestamo,
        tipo_vivienda=tipo_vivienda,
        tipo_contrato=tipo_contrato,
        edad=edad,
        renta_mensual=renta_mensual,
        gasto_por_dependiente=gasto_por_dependiente
    )

    # Preparar los datos para la predicción del modelo con los valores recibidos del formulario
    input_data = pd.DataFrame({
        'NUM_DEPENDIENTES': [num_dependientes],
        'TIEMPO_EMPLEO': [int(request.form.get('tiempo_empleo', 12))],  # Recibe desde el formulario
        'Edad': [edad],
        'Ingreso_Neto': [ingreso_mensual],
        'Score_crediticio': [int(request.form.get('score_crediticio', 633))],  # Recibe desde el formulario
        'MONTO_SOLICITADO': [monto_solicitado],
        'Plazo_del_prestamo': [meses_prestamo],
        'TIEMPO_RESIDENCIA_DOMICILIO': [int(request.form.get('tiempo_residencia_domicilio', 12))],  # Recibe desde el formulario
        'ESTADO_CIVIL_Soltero': [1 if estado_civil == 'Soltero' else 0],  # Mapeo según estado civil
        'TIPO_CONTRATO_Asalariado': [1 if tipo_contrato == 'Asalariado' else 0],
        'TIPO_DE_VIVIENDA_Propia': [1 if tipo_vivienda == 'Propia' else 0],
        'GRADO_DE_ESTUDIOS_Secundaria': [1 if grado_estudios == 'Secundaria' else 0],  # Mapeo según grado de estudios
        'Frecuencia_de_pago_Mensual': [1 if frecuencia_pago == 'Mensual' else 0]  # Mapeo según frecuencia de pago
    })

    # Alinear las columnas de input_data con las del modelo
    input_data = input_data.reindex(columns=X_train.columns, fill_value=0)
    columnas_flask = list(input_data.columns)  # Obtén las columnas del DataFrame en Flask
    columnas_xtrain = list(X_train.columns)  # Asegúrate de que X_train es el mismo usado en el entrenamiento

    set_columnas_flask = set(columnas_flask)
    set_columnas_xtrain = set(columnas_xtrain)

    print("Columnas en Flask que no están en X_train:", set_columnas_flask - set_columnas_xtrain)
    print("Columnas en X_train que no están en Flask:", set_columnas_xtrain - set_columnas_flask)
    # Ver los datos que se enviarán al modelo
    print("Datos que se enviarán al modelo:")
    print(input_data.to_string())  # Muestra todas las columnas y filas sin truncar



    # Realizar la predicción con el modelo cargado
    prediccion = regressor.predict(input_data)
    prediccion_incumplimiento = prediccion[0] * 100

    # Llamar a la función y desempaquetar el resultado
    capacidad_pago, es_aceptable = calcular_capacidad_pago(
        ingreso_mensual=ingreso_mensual,
        mensualidad=mensualidad,
        gasto_por_dependiente=gasto_por_dependiente,
        num_dependientes=num_dependientes,
        tipo_vivienda=tipo_vivienda
    )

    # Ahora puedes usar capacidad_pago y es_aceptable
    porcentaje_capacidad_pago = (capacidad_pago / ingreso_mensual) * 100


    # Calcular el score final
    # Primero calcular el score final
    # Definir un valor inicial para evitar errores
    categoria_riesgo = "Desconocido"  # O algún otro valor predeterminado
    score_final = calcular_score_final(
        porcentaje, categoria_riesgo, prediccion_incumplimiento, 
        tiempo_empleo, tiempo_residencia, 
        int(request.form.get('score_crediticio', 633))
    )

    # Ahora sí, clasificar el riesgo con el score calculado
    categoria_riesgo = clasificar_riesgo(score_final)

    # Calcular la capacidad de pago en porcentaje
    porcentaje_capacidad_pago = (capacidad_pago / ingreso_mensual) * 100

    # Decidir si el crédito es aprobado o rechazado
    if porcentaje_capacidad_pago < 30:
        decision = "Rechazado"
        mensaje_rechazo = "El crédito ha sido rechazado debido a una capacidad de pago menor al 30%."
        emoji = "😞"
        color = "#e74c3c"
    elif score_final < 600:
        decision = "Rechazado"
        mensaje_rechazo = "El crédito ha sido rechazado debido a un score final menor a 600."
        emoji = "😞"
        color = "#e74c3c"
    else:
        decision = "Aprobado"
        mensaje_aprobado = "El crédito ha sido aprobado. ¡Felicidades!"
        emoji = "😊"
        color = "#27ae60"

   

# Guardar los datos en la base de datos
    try:
        connection = get_db_connection()
        cursor = connection.cursor()

        # Generar la fecha y hora actual
        fecha_actual = datetime.now().strftime('%Y-%m-%d %H:%M:%S')  # Formato compatible con MariaDB
        id_usuario = session['user_id']  # Obtener el id del usuario

        query = """
        INSERT INTO registro_solicitud (
            nombre_cliente, ap_cliente, am_cliente, monto, mensualidad, plazo, periodicidad,
            num_dependientes, tipo_vivienda, tipo_contrato, edad, renta_mensual, gasto_por_dependiente,
            score_crediticio, porcentaje_endeudamiento, categoria_riesgo, score_final, decision, fecha, id_usuario
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
        # Imprimir consulta y valores antes de ejecutar
        #print("Consulta SQL:", query)
        #print("Valores:", (
         #   nombre, apellido_paterno, apellido_materno, monto_solicitado, mensualidad, meses_prestamo, frecuencia_pago,
          #  num_dependientes, tipo_vivienda, tipo_contrato, edad, renta_mensual, gasto_por_dependiente,
           # int(request.form.get('score_crediticio', 633)), porcentaje, categoria_riesgo, score_final, decision, fecha_actual, id_usuario
       # ))

        # Ejecutar la consulta con la fecha actual generada en Python
        cursor.execute(query, (
            nombre, apellido_paterno, apellido_materno, monto_solicitado, mensualidad, meses_prestamo, frecuencia_pago,
            num_dependientes, tipo_vivienda, tipo_contrato, edad, renta_mensual, gasto_por_dependiente,
            int(request.form.get('score_crediticio', 633)), porcentaje, categoria_riesgo, score_final, decision, fecha_actual, id_usuario  # id_usuario y fecha
        ))

        connection.commit()
        cursor.close()
        connection.close()

        flash('Los datos se han guardado correctamente en la base de datos.', 'success')

    except mariadb.Error as e:
        print(f'Error al guardar los datos: {e}')
        flash(f'Error al guardar los datos: {e}', 'danger')


    return f"""
    <html lang="es">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Resultado del Análisis de Crédito</title>
        <style>
            body {{
                font-family: 'Arial', sans-serif;
                background-color: #f4f4f9;
                margin: 0;
                padding: 0;
                color: #333;
            }}

            .container {{
                width: 80%;
                max-width: 800px;
                margin: 50px auto;
                background-color: white;
                padding: 30px;
                box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
                border-radius: 8px;
            }}

            h2 {{
                text-align: center;
                color: #2c3e50;
                margin-bottom: 20px;
            }}

            .result {{
                display: flex;
                flex-direction: column;
                gap: 15px;
                font-size: 18px;
            }}

            .result p {{
                padding: 10px;
                border-left: 5px solid #3498db;
                background-color: #ecf0f1;
                margin: 0;
                border-radius: 4px;
            }}

            .result .highlight {{
                font-weight: bold;
                font-size: 20px;
                color: #27ae60;
            }}

            .decision {{
                text-align: center;
                font-size: 22px;
                font-weight: bold;
                margin-top: 20px;
                color: {color};
            }}

            .emoji {{
                font-size: 40px;
            }}

            .footer {{
                text-align: center;
                margin-top: 30px;
                font-size: 14px;
                color: #7f8c8d;
            }}
        </style>
    </head>
    <body>
        <div class="container">
        <!-- Agregar este código dentro del body en formulario.html -->
<div style="text-align: left; margin-bottom: 20px;">
<a href="{ url_for('formulario') }" style="text-decoration: none; background-color: blue; color: white; padding: 10px 15px; border-radius: 5px;">Inicio</a>
</div>
<div style="text-align: right; margin-bottom: 20px;">
<a href="{ url_for('logout') }" style="text-decoration: none; background-color: #e74c3c; color: white; padding: 10px 15px; border-radius: 5px;">Cerrar Sesión</a>
</div>
            <div style="text-align: left; margin-bottom: 20px;">
                <img src="../static/retina-logo.png" alt="Logo" style="width: 150px; height: auto;">
            </div>
            <h2>Resultado del Análisis de Crédito</h2>
            <div class="result">
                 <p>Mensualidad aproximada: <span class="highlight">{mensualidad:.2f}</span></p>
                <p>Porcentaje de endeudamiento: <span class="highlight">{porcentaje:.2f}%</span></p>
                <p>Categoría de riesgo: <span class="highlight">{categoria_riesgo}</span></p>
                <p>Predicción de incumplimiento: <span class="highlight">{prediccion_incumplimiento:.2f}%</span></p>
                <p>Score final: <span class="highlight">{score_final:.2f}</span></p>
                 <!--<p>Capacidad de pago: <span class="highlight">{capacidad_pago:.2f}</span></p>-->
                <p>Capacidad de pago en porcentaje: <span class="highlight">{porcentaje_capacidad_pago:.2f}%</span></p>
            </div>
            <div class="decision">
                <p>Decisión: {decision} <span class="emoji">{emoji}</span></p>
                <p>
                    {mensaje_rechazo if decision == "Rechazado" else mensaje_aprobado}
                </p>
            </div>
            <div class="footer">
                <p>¡Gracias por utilizar nuestro sistema de análisis crediticio Finalia!</p>
            </div>
        </div>
    </body>
    </html>
    """
@app.route('/logout')
def logout():
    session.pop('username', None)
    flash('Has cerrado sesión exitosamente.', 'info')
    return redirect(url_for("formulario"))

if __name__ == '__main__':
    app.run(debug=True)
