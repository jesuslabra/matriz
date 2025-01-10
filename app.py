from flask import Flask, render_template, request
import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
import seaborn as sns
import joblib  # Importamos joblib para cargar el modelo

app = Flask(__name__)

# Cargar el modelo previamente entrenado
def cargar_modelo():
    # Reemplaza 'modelo.pkl' con la ruta correcta de tu modelo
    model = joblib.load('modelo.pkl')
    return model

# Cargar los datos desde el archivo CSV
data = pd.read_csv('datos.csv', encoding='latin1')

# Codificación de variables categóricas
data = pd.get_dummies(data, columns=['ESTADO_CIVIL', 'TIPO_CONTRATO', 'TIPO_DE_VIVIENDA', 
                                     'GRADO_DE_ESTUDIOS', 'Tipo_de_cargo', 'Tipo_de_ocupacion', 
                                     'Frecuencia_de_pago'], drop_first=True)

# Definir las características y la variable objetivo
features = ['NUM_DEPENDIENTES', 'TIEMPO_EMPLEO', 'Edad', 'Ingreso_Neto', 
            'Score_crediticio', 'MONTO_SOLICITADO', 'Plazo_del_prestamo', 'TIEMPO_RESIDENCIA_DOMICILIO'] + \
            list(data.columns[data.columns.str.startswith('ESTADO_CIVIL_')]) + \
            list(data.columns[data.columns.str.startswith('TIPO_CONTRATO_')]) + \
            list(data.columns[data.columns.str.startswith('TIPO_DE_VIVIENDA_')]) + \
            list(data.columns[data.columns.str.startswith('GRADO_DE_ESTUDIOS_')]) + \
            list(data.columns[data.columns.str.startswith('Tipo_de_cargo_')]) + \
            list(data.columns[data.columns.str.startswith('Tipo_de_ocupacion_')]) + \
            list(data.columns[data.columns.str.startswith('Frecuencia_de_pago')])

# Variable objetivo
target = 'Porcentaje_Incumplimiento'

# Separar las características y la variable objetivo
X = data[features]
y = data[target]

# Dividir los datos en conjunto de entrenamiento y conjunto de prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Inicializar el modelo
regressor = DecisionTreeRegressor(max_depth=7, min_samples_split=5, random_state=42)

# Realizar validación cruzada
cv_scores = cross_val_score(regressor, X_train, y_train, cv=5, scoring='neg_mean_squared_error')

# Entrenar el modelo
regressor.fit(X_train, y_train)

# Hacer predicciones
y_pred = regressor.predict(X_test)

# Función para calcular el porcentaje de endeudamiento
def calcular_porcentaje_endeudamiento(monto_solicitado, ingreso_mensual, num_dependientes, meses_prestamo, tipo_vivienda, tipo_contrato, edad, renta_mensual=None, gasto_por_dependiente=200):
    if ingreso_mensual <= 0 or meses_prestamo <= 0:
        raise ValueError("El ingreso mensual y los meses del préstamo deben ser mayores a 0.")

    carga_mensual = monto_solicitado / meses_prestamo

    if tipo_vivienda == 'inquilino' and renta_mensual is not None:
        carga_mensual += renta_mensual

    gastos_dependientes = num_dependientes * gasto_por_dependiente

    porcentaje_endeudamiento = ((carga_mensual + gastos_dependientes) / ingreso_mensual) * 100

    if tipo_contrato in ['Asalariado', 'Jubilado']:
        porcentaje_endeudamiento *= 0.85
    elif tipo_contrato in ['Trabajador Independiente', 'Trabajador Eventual']:
        porcentaje_endeudamiento *= 1.1

    if edad >= 45:
        porcentaje_endeudamiento *= 0.75
    elif edad < 25:
        porcentaje_endeudamiento *= 1.1

    if tipo_vivienda == 'propia':
        porcentaje_endeudamiento *= 0.9

    return porcentaje_endeudamiento

# Función para clasificar el riesgo crediticio
def clasificar_riesgo(puntaje_crediticio, porcentaje_endeudamiento):
    if puntaje_crediticio <= 1000:
        riesgo_continuo = max(0, (1000 - puntaje_crediticio) / 1000)
    else:
        riesgo_continuo = 1  # Riesgo extremo por puntaje inválido

    if riesgo_continuo >= 0.8:
        categoria = 'extremo'
    elif 0.6 <= riesgo_continuo < 0.8:
        categoria = 'alto'
    elif 0.4 <= riesgo_continuo < 0.6:
        categoria = 'tolerable'
    else:
        categoria = 'aceptable'

    if porcentaje_endeudamiento > 40:
        if categoria in ['aceptable', 'tolerable']:
            categoria = 'alto' if categoria == 'tolerable' else 'tolerable'
        elif categoria == 'alto':
            categoria = 'extremo'

    return categoria

# Ajustar riesgo por tiempo en empleo y residencia
def ajustar_por_estabilidad(tiempo_empleo, tiempo_residencia):
    ajuste_empleo = min(10, (tiempo_empleo - 6) / 12 * 10) if tiempo_empleo >= 6 else -10
    ajuste_residencia = min(10, tiempo_residencia / 12 * 10)
    return ajuste_empleo + ajuste_residencia

# Función para calcular el score final
def calcular_score_final(porcentaje_endeudamiento, categoria_riesgo, prediccion_incumplimiento, tiempo_empleo, tiempo_residencia, score_crediticio):
    score_endeudamiento = max(0, 100 - porcentaje_endeudamiento)
    mapping_riesgo = {
        'aceptable': 90,
        'tolerable': 70,
        'alto': 50,
        'extremo': 20
    }
    score_riesgo = mapping_riesgo.get(categoria_riesgo, 0)
    score_modelo = max(0, 100 - prediccion_incumplimiento)
    estabilidad = ajustar_por_estabilidad(tiempo_empleo, tiempo_residencia)
    bonus_score_crediticio = (score_crediticio / 1000) * 20  # Hasta 20 puntos adicionales por score alto

    score_final = (
        0.34 * score_endeudamiento + 
        0.29 * score_riesgo + 
        0.12 * score_modelo + 
        0.09 * estabilidad + 
        0.20 * bonus_score_crediticio
    )

    return score_final


# Ruta para mostrar el formulario
@app.route("/", methods=["GET"])
def formulario():
    return render_template("formulario.html")

# Ruta para recibir los datos y hacer el análisis
@app.route("/calcular_credito", methods=["POST"])
def calcular_credito():
    # Obtener los datos del formulario
    monto_solicitado = float(request.form["monto_solicitado"])
    ingreso_mensual = float(request.form["ingreso_mensual"])
    num_dependientes = int(request.form["num_dependientes"])
    meses_prestamo = int(request.form["meses_prestamo"])
    tipo_vivienda = request.form["tipo_vivienda"]
    tipo_contrato = request.form["tipo_contrato"]
    edad = int(request.form["edad"])
    renta_mensual = float(request.form["renta_mensual"]) if request.form["renta_mensual"] else 0
    gasto_por_dependiente = float(request.form["gasto_por_dependiente"])
    score_crediticio = int(request.form["score_crediticio"])
    tiempo_empleo = int(request.form["tiempo_empleo"])
    tiempo_residencia = int(request.form["tiempo_residencia"])

    # Cargar el modelo previamente entrenado
    regressor = cargar_modelo()

    porcentaje = calcular_porcentaje_endeudamiento(
        monto_solicitado, ingreso_mensual, num_dependientes, meses_prestamo,
        tipo_vivienda, tipo_contrato, edad, renta_mensual, gasto_por_dependiente
    )
    
    # Preparar los datos de entrada
    input_data = pd.DataFrame({
        'NUM_DEPENDIENTES': [num_dependientes],
        'TIEMPO_EMPLEO': [tiempo_empleo],
        'Edad': [edad],
        'Ingreso_Neto': [ingreso_mensual],
        'Score_crediticio': [score_crediticio],
        'MONTO_SOLICITADO': [monto_solicitado],
        'Plazo_del_prestamo': [meses_prestamo],
        'TIEMPO_RESIDENCIA_DOMICILIO': [tiempo_residencia],
        'ESTADO_CIVIL_Jubilado': [0],  # Asumiendo que no es jubilado
        'TIPO_CONTRATO_Asalariado': [1],  # Ejemplo de tipo de contrato
        'TIPO_DE_VIVIENDA_Propia': [1],  # Ejemplo de tipo de vivienda
        'GRADO_DE_ESTUDIOS_Secundaria': [0],  # Ejemplo de grado de estudios
        'Tipo_de_cargo_Administrativo': [0],  # Ejemplo de tipo de cargo
        'Tipo_de_ocupacion_Empleado': [1],  # Ejemplo de ocupación
        'Frecuencia_de_pago_Mensual': [0]  # Ejemplo de frecuencia de pago
    })

    # Alinear las columnas para que coincidan con las que se usaron en el entrenamiento
    columns = X_train.columns
    input_data = input_data.reindex(columns=columns, fill_value=0)

    # Hacer la predicción
    prediccion = regressor.predict(input_data)
    prediccion_incumplimiento = prediccion[0] * 100

    # Calcular el score final
    categoria_riesgo = clasificar_riesgo(score_crediticio, porcentaje)
    score_final = calcular_score_final(porcentaje, categoria_riesgo, prediccion_incumplimiento, tiempo_empleo, tiempo_residencia, score_crediticio)

    # Decidir si el crédito es aprobado o rechazado
    decision = "Aprobado" if score_final >= 60 else "Rechazado"
    # Definir la carita según la decisión
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
            <h2>Resultado del Análisis de Crédito</h2>
            <div class="result">
                <p>El porcentaje de endeudamiento es: <span class="highlight">{porcentaje:.2f}%</span></p>
                <p>La predicción del incumplimiento es: <span class="highlight">{prediccion_incumplimiento:.2f}%</span></p>
                <p>El score final es: <span class="highlight">{score_final:.2f}</span></p>
            </div>
            <div class="decision">
                <p>Decisión: {decision} <span class="emoji">{emoji}</span></p>
            </div>
            <div class="footer">
                <p>¡Gracias por utilizar nuestro sistema de análisis crediticio!</p>
            </div>
        </div>
    </body>
    </html>
    """


if __name__ == "__main__":
    app.run(debug=True)
