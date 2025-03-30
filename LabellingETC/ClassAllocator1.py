import os

label_dir = r"C:\Users\User\Desktop\DATAV4\train\labels"

# Define class mappings (old -> new)
class_mapping = {4: }  # Change '2' (old person/surfer class) to '1'

for filename in os.listdir(label_dir):
    if filename.endswith(".txt"):
        file_path = os.path.join(label_dir, filename)

        with open(file_path, "r") as file:
            lines = file.readlines()

        new_lines = []
        for line in lines:
            parts = line.split()
            class_id = int(parts[0])

            # Update class ID if in mapping
            if class_id in class_mapping:
                parts[0] = str(class_mapping[class_id])

            new_lines.append(" ".join(parts) + "\n")

        # Overwrite with updated labels
        with open(file_path, "w") as file:
            file.writelines(new_lines)

print("Label realignment complete!")
