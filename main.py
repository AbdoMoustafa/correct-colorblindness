import os
import process_image
import refinements
import colorama
from colorama import Fore, Back, Style


def main():
    colorama.init(autoreset=True)
    deficiency = refinements.get_deficiency_input()
    user_choice = refinements.get_user_input()
    results_folder = "results"

    if not os.path.exists(results_folder):
        os.makedirs(results_folder)

    if user_choice == '1':
        file_path = input("\n" + Fore.MAGENTA + Style.BRIGHT + Back.BLACK + " Enter the path to the image: ").strip()
        process_image.process_image(file_path, results_folder, deficiency, user_choice)
    elif user_choice == '2':
        folder_path = input("\n" + Fore.MAGENTA + Style.BRIGHT + Back.BLACK + " Enter the path to the folder: ").strip()
        for file_name in os.listdir(folder_path):
            try:
                if file_name.endswith('.png') or file_name.endswith('.jpg'):
                    file_path = os.path.join(folder_path, file_name)
                    process_image.process_image(file_path, results_folder, deficiency, user_choice)
                else:
                    print(Fore.LIGHTYELLOW_EX + Style.BRIGHT + f"Skipping non-image file: {file_name}")

            except Exception as e:
                continue
        print(Style.BRIGHT + Back.RESET + Fore.GREEN + f"\n\nYour new images has been processed and saved to: {results_folder}/")
    else:
        print(Fore.LIGHTYELLOW_EX + Style.BRIGHT + "Invalid choice. Please enter '1' or '2'.")


if __name__ == "__main__":
    main()
