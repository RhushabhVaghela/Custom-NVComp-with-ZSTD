import sys

def main():
    if len(sys.argv) < 2:
        print("Usage: python3 dump_hex.py <file>")
        return

    filename = sys.argv[1]
    with open(filename, 'rb') as f:
        data = f.read()
        print(f"File: {filename}, Size: {len(data)}")
        for i in range(0, len(data), 16):
            chunk = data[i:i+16]
            hex_str = ' '.join(f"{b:02X}" for b in chunk)
            print(f"{i:04X}: {hex_str}")

if __name__ == "__main__":
    main()
