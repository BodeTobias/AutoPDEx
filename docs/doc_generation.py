import os
import re

# Funktion zum Extrahieren der Gruppierungen und Funktionen
def extract_groups(filename):
    groups = {}
    current_group = None
    group_found = False

    # Öffnen der Python-Datei mit UTF-8-Kodierung und Durchsuchen nach Kommentaren und Funktionen
    with open(filename, 'r', encoding='utf-8') as file:
        for line in file:
            # Sucht nach Gruppierungen wie ### Group 1
            group_match = re.match(r'### (.*)', line)
            if group_match:
                group_found = True
                current_group = group_match.group(1)
                groups[current_group] = []
            # Sucht nach Funktionsdefinitionen, die nicht mit einem _ anfangen
            func_match = re.match(r'def ([a-zA-Z_]\w*)\(', line)
            if func_match:
                func_name = func_match.group(1)
                if not func_name.startswith('_'):  # Funktionen mit _ ignorieren
                    if current_group:
                        groups[current_group].append(func_name)
                    else:
                        if None not in groups:
                            groups[None] = []
                        groups[None].append(func_name)
    return groups, group_found

# Funktion zum Erstellen des RST-Inhalts
def generate_rst(groups, module_name, group_found):
    rst_output = []
    # Beginne mit einem Header
    header = f"{module_name.replace('_', ' ').capitalize()}\n{'=' * len(module_name)}\n\n"
    rst_output.append(header)
    rst_output.append(f".. automodule:: autopdex.{module_name}\n    :no-index:\n\n")

    # Falls Gruppierungen vorhanden sind
    if group_found:
        for group, functions in groups.items():
            if group:  # Nur Gruppen mit Namen anzeigen
                rst_output.append(f'{group}\n{"-" * len(group)}\n')
                rst_output.append('.. autosummary::\n   :toctree: _autosummary\n\n')
                for func in functions:
                    rst_output.append(f'   {func}\n')
                rst_output.append('\n')
    else:
        # Wenn keine Gruppierungen vorhanden sind, einfach alle Funktionen auflisten
        rst_output.append('.. autosummary::\n   :toctree: _autosummary\n\n')
        for func in groups.get(None, []):
            rst_output.append(f'   {func}\n')
        rst_output.append('\n')
    
    return ''.join(rst_output)

# Hauptfunktion zum Durchführen des Skripts
if __name__ == "__main__":
    # Liste der Module, für die RST-Dateien generiert werden sollen
    target_modules = ['assembler', 'geometry', 'mesher', 'models', 'plotter', 'seeder', 'solution_structures', 'spaces', 'utility']  # Beispiel-Liste der Module

    # Ordner, in dem sich die Python-Dateien befinden
    python_files_dir = './autopdex'

    # Ordner, in den die RST-Dateien geschrieben werden sollen
    output_dir = './docs/generated_rst'

    # Überprüfen, ob der Ordner existiert, wenn nicht, erstellen
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Überprüfen, ob der Ordner mit den Python-Dateien existiert
    if not os.path.exists(python_files_dir):
        print(f"Ordner {python_files_dir} existiert nicht.")
        exit(1)

    # Iteriere durch alle angegebenen Python-Dateien in der Liste 'target_modules'
    for module_name in target_modules:
        filename = f'{module_name}.py'
        
        # Erlaubt Dateinamen mit Unterstrichen (_) und verarbeitet sie
        filepath = os.path.join(python_files_dir, filename)
        
        # Überprüfen, ob die Datei existiert
        if os.path.exists(filepath):
            groups, group_found = extract_groups(filepath)
            rst_content = generate_rst(groups, module_name, group_found)
            
            # Speichere die Ausgabe als RST-Datei in 'generated_rst'
            output_rst = os.path.join(output_dir, f'{module_name}.rst')
            with open(output_rst, 'w', encoding='utf-8') as rst_file:
                rst_file.write(rst_content)
            
            print(f'RST-Datei für {filename} wurde erstellt: {output_rst}')
        else:
            print(f"Datei {filename} existiert nicht im Ordner {python_files_dir}.")