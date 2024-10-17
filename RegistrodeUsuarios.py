import json
import re
import tkinter as tk
from tkinter import messagebox, simpledialog

def leer_datos(ruta_archivo):
    try:
        with open(ruta_archivo, 'r') as archivo:
            return json.load(archivo)
    except FileNotFoundError:
        return []

def escribir_datos(datos, ruta_archivo):
    with open(ruta_archivo, 'w') as archivo:
        json.dump(datos, archivo, indent=2)

def es_correo_valido(direccion):
    patron = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return re.match(patron, direccion) is not None

def registrar_entrada(lista_datos):
    nombre = simpledialog.askstring("Registrar", "Nombre:").strip()
    tel = simpledialog.askstring("Registrar", "Teléfono:").strip()
    
    while True:
        correo = simpledialog.askstring("Registrar", "Correo electrónico:").strip()
        if es_correo_valido(correo):
            break
        messagebox.showerror("Error", "Correo electrónico inválido. Inténtelo de nuevo.")
    
    entrada = {
        'nombre': nombre,
        'tel': tel,
        'correo': correo
    }
    
    lista_datos.append(entrada)
    messagebox.showinfo("Registro Exitoso", f"Entrada para {nombre} registrada exitosamente.")

def listar_entradas(lista_datos):
    if not lista_datos:
        messagebox.showinfo("Listar Entradas", "No hay entradas registradas.")
        return
    
    lista = "\n".join([f"Nombre: {e['nombre']}, Tel: {e['tel']}, Correo: {e['correo']}" for e in lista_datos])
    messagebox.showinfo("Listar Entradas", lista)

def buscar_entrada(lista_datos):
    busqueda = simpledialog.askstring("Buscar Entrada", "Ingrese el nombre a buscar:").lower()
    resultados = [e for e in lista_datos if busqueda in e['nombre'].lower()]
    
    if resultados:
        lista = "\n".join([f"Nombre: {e['nombre']}, Tel: {e['tel']}, Correo: {e['correo']}" for e in resultados])
        messagebox.showinfo("Resultados de la Búsqueda", lista)
    else:
        messagebox.showinfo("Sin Coincidencias", "No se encontraron coincidencias.")

def borrar_entrada(lista_datos):
    nombre = simpledialog.askstring("Borrar Entrada", "Ingrese el nombre de la entrada a borrar:")
    indice = next((i for i, e in enumerate(lista_datos) if e['nombre'].lower() == nombre.lower()), None)
    
    if indice is not None:
        borrado = lista_datos.pop(indice)
        messagebox.showinfo("Entrada Eliminada", f"Entrada de {borrado['nombre']} eliminada.")
    else:
        messagebox.showerror("Error", "Entrada no encontrada.")

def menu_principal():
    ruta_archivo = 'datos.json'
    lista_datos = leer_datos(ruta_archivo)
    
    def registrar():
        registrar_entrada(lista_datos)
    
    def listar():
        listar_entradas(lista_datos)
    
    def buscar():
        buscar_entrada(lista_datos)
    
    def borrar():
        borrar_entrada(lista_datos)
    
    def salir():
        escribir_datos(lista_datos, ruta_archivo)
        root.quit()
    
    root = tk.Tk()
    root.title("Gestor de Entradas")
    
    tk.Button(root, text="Registrar Nueva Entrada", command=registrar).pack(fill=tk.X)
    tk.Button(root, text="Listar Todas las Entradas", command=listar).pack(fill=tk.X)
    tk.Button(root, text="Buscar Entrada", command=buscar).pack(fill=tk.X)
    tk.Button(root, text="Borrar Entrada", command=borrar).pack(fill=tk.X)
    tk.Button(root, text="Salir", command=salir).pack(fill=tk.X)
    
    root.mainloop()

if __name__ == "__main__":
    menu_principal()
