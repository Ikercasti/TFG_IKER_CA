import serial
import pyqtgraph as pg
from PyQt5.QtWidgets import QApplication
from pyqtgraph.Qt import QtCore
import numpy as np
import csv

# --- Configuración ---
puerto = 'COM13'
baudios = 921600
muestras = 3000
v_max = 3.3
intervalo_us = 100  # ESP32 → 100 µs

# --- Serial ---
ser = serial.Serial(puerto, baudios, timeout=1)

# --- Gráfica ---
app = QApplication([])
win = pg.GraphicsLayoutWidget(title="Señales desde ESP32")
win.show()
plot = win.addPlot(title="ADC ESP32:")
tiempo = np.linspace(-((muestras - 1) * intervalo_us / 1e6), 0, muestras)
curve1 = plot.plot(tiempo, np.zeros(muestras), pen='w')  # Blanco
curve2 = plot.plot(tiempo, np.zeros(muestras), pen='y')  # Amarillo

plot.setLabel('left', 'Voltaje', units='V')
plot.setLabel('bottom', 'Tiempo', units='s')
plot.setYRange(0, v_max)

# --- Datos ---
data1 = np.zeros(muestras)
data2 = np.zeros(muestras)
raw_values1, raw_values2 = [], []
labels = []

current_label = None  # Etiqueta actual

def update():
    global data1, data2, raw_values1, raw_values2, labels, current_label
    count = 0
    max_reads = 200
    while ser.in_waiting and count < max_reads:
        try:
            line = ser.readline().decode().strip()
            if ',' not in line:
                continue
            val1_str, val2_str = line.split(',')
            val1 = int(val1_str)
            val2 = int(val2_str)
            volt1 = (val1 * v_max) / 4095.0
            volt2 = (val2 * v_max) / 4095.0

            data1 = np.roll(data1, -1)
            data2 = np.roll(data2, -1)
            data1[-1] = volt1
            data2[-1] = volt2

            raw_values1.append(volt1)
            raw_values2.append(volt2)
            labels.append(current_label if current_label is not None else "ninguna")

            count += 1
        except Exception as e:
            print(f"Error: {e}")

    curve1.setData(tiempo, data1)
    curve2.setData(tiempo, data2)

# --- Temporizador ---
timer = QtCore.QTimer()
timer.timeout.connect(update)
timer.start(10)

# --- Guardar datos al salir ---
def guardar_csv():
    with open('prueba.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=';')
        writer.writerow(['Voltaje 1 (V)', 'Voltaje 2 (V)', 'Etiqueta'])
        for v1, v2, label in zip(raw_values1, raw_values2, labels):
            writer.writerow([v1, v2, label])
    print("Datos guardados en prueba_doble.csv")

app.aboutToQuit.connect(guardar_csv)

# --- Manejo de teclado para etiquetas ---
def key_press(event):
    global current_label
    key = event.text()
    if key == '1':
        current_label = 'golpe_dedo'
    elif key == '2':
        current_label = 'caricia_dedo'
    elif key == '3':
        current_label = 'golpe_mano'
    elif key == '4':
        current_label = 'caricia_mano'
    elif key == 'q':
        current_label = None
    print(f"Etiqueta actual: {current_label}")

app.focusChanged.connect(lambda old, new: app.installEventFilter(win))
win.keyPressEvent = key_press

# --- Ejecutar ---
app.exec_()
