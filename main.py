"""
Script principal para ejecutar todos los ejercicios del TP2
"""
import sys
sys.path.insert(0, '/home/lu/Escritorio/tp2-redes-neuronales')

def main():
    """Menú principal para ejecutar los ejercicios"""
    
    print("=" * 70)
    print("TRABAJO PRÁCTICO 2: REDES NEURONALES")
    print("=" * 70)
    print()
    print("Seleccione un ejercicio para ejecutar:")
    print()
    print("1. Perceptrón Simple (AND/OR, 2 y 4 entradas)")
    print("2. Capacidad del Perceptrón")
    print("3. Perceptrón Multicapa (XOR - Backpropagation)")
    print("4. Red Backpropagation (sin(x)+cos(y)+z)")
    print("5. Máquina Restringida de Boltzmann (MNIST)")
    print("6. Red Convolucional (MNIST)")
    print("7. Autoencoder (MNIST)")
    print("8. XOR con Simulated Annealing")
    print("9. XOR con Algoritmo Genético")
    print("0. Salir")
    print()
    
    while True:
        try:
            choice = input("Ingrese su opción (0-9): ").strip()
            
            if choice == '0':
                print("¡Hasta luego!")
                break
            
            elif choice == '1':
                print("\nEjecutando Ejercicio 1...")
                from ejercicios import ejercicio1
                ejercicio1.ejercicio1()
            
            elif choice == '2':
                print("\nEjecutando Ejercicio 2...")
                from ejercicios import ejercicio2
                ejercicio2.ejercicio2()
            
            elif choice == '3':
                print("\nEjecutando Ejercicio 3...")
                from ejercicios import ejercicio3
                ejercicio3.ejercicio3()
            
            elif choice == '4':
                print("\nEjecutando Ejercicio 4...")
                from ejercicios import ejercicio4
                print("\nParte 4a: Red para f(x,y,z)=sin(x)+cos(y)+z")
                ejercicio4.ejercicio4a()
                print("\nParte 4b: Impacto del minibatch")
                ejercicio4.ejercicio4b()
            
            elif choice == '5':
                print("\nEjecutando Ejercicio 5...")
                from ejercicios import ejercicio5
                ejercicio5.ejercicio5()
            
            elif choice == '6':
                print("\nEjecutando Ejercicio 6...")
                from ejercicios import ejercicio6
                ejercicio6.ejercicio6()
            
            elif choice == '7':
                print("\nEjecutando Ejercicio 7...")
                from ejercicios import ejercicio7
                ejercicio7.ejercicio7()
            
            elif choice == '8':
                print("\nEjecutando Ejercicio 8...")
                from ejercicios import ejercicio8
                ejercicio8.ejercicio8()
            
            elif choice == '9':
                print("\nEjecutando Ejercicio 9...")
                from ejercicios import ejercicio9
                print("\nParte 9a: XOR con Algoritmo Genético")
                ejercicio9.ejercicio9a()
                print("\nParte 9b: Impacto de parámetros")
                ejercicio9.ejercicio9b()
            
            else:
                print("Opción no válida. Intente de nuevo.")
        
        except Exception as e:
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()
        
        print()


if __name__ == "__main__":
    main()
