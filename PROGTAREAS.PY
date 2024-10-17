import json
import re
import datetime

class Tarea:
    def __init__(self, titulo, descripcion, fecha_vencimiento):
        self.titulo = titulo
        self.descripcion = descripcion
        self.fecha_vencimiento = fecha_vencimiento
        self.completada = False

    def marcar_completada(self):
        self.completada = True

    def to_dict(self):
        return {
            'titulo': self.titulo,
            'descripcion': self.descripcion,
            'fecha_vencimiento': self.fecha_vencimiento.strftime('%Y-%m-%d'),
            'completada': self.completada
        }

    @staticmethod
    def from_dict(data):
        fecha_vencimiento = datetime.datetime.strptime(data['fecha_vencimiento'], '%Y-%m-%d')
        tarea = Tarea(data['titulo'], data['descripcion'], fecha_vencimiento)
        tarea.completada = data['completada']
        return tarea
    
class SistemaDeGestionDeTareas:
    def __init__(self, archivo='tareas.json'):
        self.archivo = archivo
        self.tareas = self.cargar_tareas()

    def agregar_tarea(self, titulo, descripcion, fecha_vencimiento):
        nueva_tarea = Tarea(titulo, descripcion, fecha_vencimiento)
        self.tareas.append(nueva_tarea)
        print("Tarea agregada exitosamente.")

    def mostrar_tareas(self):
        if not self.tareas:
            print("No hay tareas pendientes.")
            return

        tareas_ordenadas = sorted(self.tareas, key=lambda x: x.fecha_vencimiento)
        for idx, tarea in enumerate(tareas_ordenadas, 1):
            estado = 'Completada' if tarea.completada else 'Pendiente'
            print(f"{idx}. Título: {tarea.titulo}, Descripción: {tarea.descripcion}, Fecha de vencimiento: {tarea.fecha_vencimiento.strftime('%Y-%m-%d')}, Estado: {estado}")

    def marcar_tarea_completada(self, titulo):
        for tarea in self.tareas:
            if tarea.titulo.lower() == titulo.lower():
                tarea.marcar_completada()
                print("Tarea marcada como completada.")
                return
        print("Tarea no encontrada.")

    def eliminar_tarea(self, titulo):
        for tarea in self.tareas:
            if tarea.titulo.lower() == titulo.lower():
                self.tareas.remove(tarea)
                print("Tarea eliminada.")
                return
        print("Tarea no encontrada.")

    def buscar_tareas(self, termino):
        resultados = [tarea for tarea in self.tareas if termino.lower() in tarea.titulo.lower() or termino.lower() in tarea.descripcion.lower()]
        if resultados:
            for tarea in resultados:
                estado = 'Completada' if tarea.completada else 'Pendiente'
                print(f"Título: {tarea.titulo}, Descripción: {tarea.descripcion}, Fecha de vencimiento: {tarea.fecha_vencimiento.strftime('%Y-%m-%d')}, Estado: {estado}")
        else:
            print("No se encontraron tareas.")

    def guardar_tareas(self):
        with open(self.archivo, 'w') as archivo:
            json.dump([tarea.to_dict() for tarea in self.tareas], archivo)
        print("Tareas guardadas exitosamente.")

    def cargar_tareas(self):
        try:
            with open(self.archivo, 'r') as archivo:
                tareas_data = json.load(archivo)
                return [Tarea.from_dict(data) for data in tareas_data]
        except FileNotFoundError:
            return []

    def run(self):
        while True:
            print("\n--- Menú ---")
            print("1. Agregar Tarea")
            print("2. Mostrar Tareas")
            print("3. Marcar Tarea como Completada")
            print("4. Eliminar Tarea")
            print("5. Buscar Tareas")
            print("6. Guardar y Salir")
            print("7. Salir sin Guardar")

            opcion = input("Seleccione una opción: ")

            if opcion == '1':
                titulo = input("Ingrese el título de la tarea: ")
                descripcion = input("Ingrese la descripción de la tarea: ")
                fecha_vencimiento = input("Ingrese la fecha de vencimiento (YYYY-MM-DD): ")
                fecha_vencimiento = datetime.datetime.strptime(fecha_vencimiento, '%Y-%m-%d')
                self.agregar_tarea(titulo, descripcion, fecha_vencimiento)
            elif opcion == '2':
                self.mostrar_tareas()
            elif opcion == '3':
                titulo = input("Ingrese el título de la tarea a marcar como completada: ")
                self.marcar_tarea_completada(titulo)
            elif opcion == '4':
                titulo = input("Ingrese el título de la tarea a eliminar: ")
                self.eliminar_tarea(titulo)
            elif opcion == '5':
                termino = input("Ingrese un término para buscar tareas: ")
                self.buscar_tareas(termino)
            elif opcion == '6':
                self.guardar_tareas()
                print("Saliendo del programa.")
                break
            elif opcion == '7':
                print("Saliendo sin guardar.")
                break
            else:
                print("Opción no válida. Intente nuevamente.")

if __name__ == "__main__":
    sistema = SistemaDeGestionDeTareas()
    sistema.run()