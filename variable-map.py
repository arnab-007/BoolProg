import json
my_list = [];

with open('var-mapping') as f:
    lines = f.readlines() # list containing lines of file
    columns = [] # To store column names


map = lines[[ i for i, word in enumerate(lines) if word.startswith('c ') ][0]:]


variable_mapping = {}
guard_mapping = {}

for line in lines:
    parts = line.split(' ')
    if line.startswith('c main::'):
        variable_parts = parts[1].split('::')[2].split('!')

for line in map:
    parts = line.split()
    if line.startswith('c goto_symex::\\guard#'):
      guard_number = parts[1].split('::')[1].split('\\')[1]
      guard_mapping[f'{guard_number}'] = int(parts[-1])
    elif line.startswith('c main::'):
      variable_parts = parts[1].split('::')[2].split('!') # Extract variable name
      variable_name, instance = variable_parts[0],variable_parts[1].split('#')[1]
      variable_name = f"{variable_name}_{instance}"
      variable_mapping[f"{variable_name}"] = int(parts[-1])

# Print guard mapping
for guard, value in guard_mapping.items():
    print(f"{guard}: {value}")

# Print variable mapping
for variable, location in variable_mapping.items():
    print(f"{variable}: {location}")