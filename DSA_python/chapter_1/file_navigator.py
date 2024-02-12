def file_reader(file_path):
    file_list = []
    with open(file_path, 'r') as file:
        for line in file:
            file_list.append(line.strip())
    return file_list


file_path = input('Enter the file path: ')
file_list = file_reader(file_path)
line_number = input('Enter the line number: ')
if line_number.isdigit() and int(line_number) > 0 and int(line_number) <= len(file_list):
    print(file_list[int(line_number) - 1])
else: 
    print("Invalid line number. Please enter a number between 1 and the number of lines in the file.")