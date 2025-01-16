from flask import Flask, render_template, request
import pandas as pd
import joblib

# Inicialización de la aplicación Flask
app = Flask(__name__)

# Cargar el modelo previamente guardado
regressor = joblib.load('modelo_crediticio.pkl')

# Función para calcular el porcentaje de endeudamiento
def calcular_porcentaje_endeudamiento(monto_solicitado, ingreso_mensual, num_dependientes, meses_prestamo, tipo_vivienda, tipo_contrato, edad, renta_mensual, gasto_por_dependiente):
    # Lógica para calcular la mensualidad y el porcentaje de endeudamiento
    mensualidad = monto_solicitado / meses_prestamo
    gastos_dependientes = gasto_por_dependiente * num_dependientes
    total_gastos = mensualidad + gastos_dependientes + renta_mensual
    porcentaje = (total_gastos / ingreso_mensual) * 100
    return mensualidad, porcentaje

# Función para calcular la capacidad de pago
def calcular_capacidad_pago(ingreso_mensual, mensualidad, gasto_por_dependiente, num_dependientes, tipo_vivienda):
    # Lógica para calcular la capacidad de pago
    gastos_dependientes = gasto_por_dependiente * num_dependientes
    total_gastos = mensualidad + gastos_dependientes
    capacidad_pago = ingreso_mensual - total_gastos
    return capacidad_pago

# Función para clasificar el riesgo basado en el porcentaje
def clasificar_riesgo(score_crediticio, porcentaje):
    # Clasificación del riesgo según el porcentaje
    if porcentaje < 30:
        return 'Bajo'
    elif porcentaje < 50:
        return 'Medio'
    else:
        return 'Alto'

# Función para calcular el score final
def calcular_score_final(porcentaje, categoria_riesgo, prediccion_incumplimiento, tiempo_empleo, tiempo_residencia, score_crediticio):
    # Lógica para calcular el score final
    score_final = score_crediticio - (porcentaje * 0.5) - (prediccion_incumplimiento * 0.5)
    return score_final

@app.route('/')
def index():
    return render_template('formulario.html')

@app.route('/calcular_credito', methods=['POST'])
def calcular_credito():
    # Recibir los datos del formulario
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
    input_data = input_data.reindex(columns=regressor.feature_names_in_, fill_value=0)

    # Realizar la predicción
    prediccion = regressor.predict(input_data)
    prediccion_incumplimiento = prediccion[0] * 100

    # Calcular la capacidad de pago
    capacidad_pago = calcular_capacidad_pago(
        ingreso_mensual=ingreso_mensual,
        mensualidad=mensualidad,
        gasto_por_dependiente=gasto_por_dependiente,
        num_dependientes=num_dependientes,
        tipo_vivienda=tipo_vivienda
    )

    # Calcular el score final
    categoria_riesgo = clasificar_riesgo(int(request.form.get('score_crediticio', 633)), porcentaje)
    score_final = calcular_score_final(porcentaje, categoria_riesgo, prediccion_incumplimiento, 12, 12, 633)

    # Calcular la capacidad de pago en porcentaje
    porcentaje_capacidad_pago = (capacidad_pago / ingreso_mensual) * 100

    # Decidir si el crédito es aprobado o rechazado
    if porcentaje_capacidad_pago < 30:
        decision = "Rechazado"
        emoji = "😞"
        color = "#e74c3c"
    else:
        decision = "Aprobado" if score_final >= 60 else "Rechazado"
        emoji = "😊" if decision == "Aprobado" else "😞"
        color = "#27ae60" if decision == "Aprobado" else "#e74c3c"

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
            <div style="text-align: left; margin-bottom: 20px;">
                <img src="../static/retina-logo.png" alt="Logo" style="width: 150px; height: auto;">
            </div>
            <h2>Resultado del Análisis de Crédito</h2>
            <div class="result">
                <p>Mensualidad aproximada: <span class="highlight">{mensualidad:.2f}</span></p>
                <p>Porcentaje de endeudamiento: <span class="highlight">{porcentaje:.2f}%</span></p>
                <p>Categoría de riesgo: <span class="highlight">{categoria_riesgo}</span></p>
                <!--<p>Predicción de incumplimiento: <span class="highlight">{prediccion_incumplimiento:.2f}%</span></p>-->
                <p>Score final: <span class="highlight">{score_final:.2f}</span></p>
                 <!--<p>Capacidad de pago: <span class="highlight">{capacidad_pago:.2f}</span></p>-->
                <p>Capacidad de pago: <span class="highlight">{porcentaje_capacidad_pago:.2f}%</span></p>
            </div>
            <div class="decision">
                <p>Decisión: {decision} <span class="emoji">{emoji}</span></p>
            </div>
            <div class="footer">
                <p>¡Gracias por utilizar nuestro sistema de análisis crediticio Finalia!</p>
            </div>
        </div>
    </body>
    </html>
    """

if __name__ == '__main__':
    app.run(debug=True)
