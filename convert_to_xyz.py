import re

def extract_xyz_frames(input_file, output_file):
    with open(input_file, 'r') as f:
        content = f.read()

    # This regex looks for the header and then captures all consecutive "atom" lines
    # It stops when it hits the horizontal dashed line or a line without "atom"
    block_pattern = re.compile(
        r"Atomic structure that was used in the preceding time step of the wrapper.*?\n"
        r"(?:\s+x \[A\]\s+y \[A\]\s+z \[A\]\n)"
        r"((?:\s+atom\s+[\d.-]+\s+[\d.-]+\s+[\d.-]+\s+[A-Za-z]+\n)+)",
        re.MULTILINE
    )

    blocks = block_pattern.findall(content)
    
    with open(output_file, 'w') as out:
        for i, block in enumerate(blocks):
            lines = block.strip().split('\n')
            atom_count = len(lines)
            
            # Write XYZ header
            out.write(f"{atom_count}\n")
            out.write(f"Frame {i}: Extracted from log file\n")
            
            for line in lines:
                parts = line.split()
                # parts[0] is "atom"
                # parts[1, 2, 3] are x, y, z
                # parts[4] is the Element symbol
                x, y, z, element = parts[1], parts[2], parts[3], parts[4]
                out.write(f"{element:<3} {x:>15} {y:>15} {z:>15}\n")

    print(f"Extraction complete. {len(blocks)} frames written to {output_file}.")

# Usage
extract_xyz_frames('aims.out', 'trajectory.xyz')