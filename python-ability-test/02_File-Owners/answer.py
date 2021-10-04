def group_by_owners(files):
    out = {}
    for key, value in files.items():
        if value in out:
            out[value].append(key)
        else:
            out[value] = [key]
    return out

if __name__ == "__main__":    
    files = {
        'Input.txt': 'Randy',
        'Code.py': 'Stan',
        'Output.txt': 'Randy'
    }   
    print(group_by_owners(files))