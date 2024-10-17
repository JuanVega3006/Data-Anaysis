import json
import re

# Función para cargar contactos desde un archivo
def cargar_contactos(nombre_archivo):
    try:
        with open(nombre_archivo, 'r') as archivo:
            return json.load(archivo)
    except FileNotFoundError:
        return []

# Función para guardar contactos en un archivo
def guardar_contactos(contactos, nombre_archivo):
    with open(nombre_archivo, 'w') as archivo:
        json.dump(contactos, archivo)

# Función para agregar un contacto
def agregar_contacto(contactos):
    nombre = input("Ingrese el nombre: ")
    telefono = input("Ingrese el número de teléfono: ")
    email = input("Ingrese el correo electrónico: ")
    
    # Validar correo electrónico
    if not re.match(r"[^@]+@[^@]+\.[^@]+", email):
        print("Correo electrónico no válido.")
        return
    
    contactos.append({"nombre": nombre, "telefono": telefono, "email": email})
    print("Contacto agregado exitosamente.")

# Función para mostrar todos los contactos
def mostrar_contactos(contactos):
    if not contactos:
        print("No hay contactos en la lista.")
        return

    for idx, contacto in enumerate(contactos, 1):
        print(f"{idx}. Nombre: {contacto['nombre']}, Teléfono: {contacto['telefono']}, Email: {contacto['email']}")

# Función para buscar un contacto por nombre
def buscar_contacto(contactos):
    nombre = input("Ingrese el nombre del contacto a buscar: ")
    for contacto in contactos:
        if contacto['nombre'].lower() == nombre.lower():
            print(f"Nombre: {contacto['nombre']}, Teléfono: {contacto['telefono']}, Email: {contacto['email']}")
            return
    print("Contacto no encontrado.")

# Función para eliminar un contacto por nombre
def eliminar_contacto(contactos):
    nombre = input("Ingrese el nombre del contacto a eliminar: ")
    for contacto in contactos:
        if contacto['nombre'].lower() == nombre.lower():
            contactos.remove(contacto)
            print("Contacto eliminado exitosamente.")
            return
    print("Contacto no encontrado.")

# Función principal
def main():
    nombre_archivo = "contactos.json"
    contactos = cargar_contactos(nombre_archivo)
    
    while True:
        print("\n--- Menú ---")
        print("1. Agregar Contacto")
        print("2. Mostrar Contactos")
        print("3. Buscar Contacto")
        print("4. Eliminar Contacto")
        print("5. Guardar y Salir")
        print("6. Salir sin Guardar")
        
        opcion = input("Seleccione una opción: ")
        
        if opcion == '1':
            agregar_contacto(contactos)
        elif opcion == '2':
            mostrar_contactos(contactos)
        elif opcion == '3':
            buscar_contacto(contactos)
        elif opcion == '4':
            eliminar_contacto(contactos)
        elif opcion == '5':
            guardar_contactos(contactos, nombre_archivo)
            print("Contactos guardados. Saliendo del programa.")
            break
        elif opcion == '6':
            print("Saliendo sin guardar.")
            break
        else:
            print("Opción no válida. Intente nuevamente.")

if __name__ == "__main__":
           main()
   