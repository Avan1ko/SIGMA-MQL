"""
Generate preview images for all map types (SIGMA and MQL)
"""

import os
import matplotlib.pyplot as plt
from map_generator import MapGenerator, MapType


def generate_all_previews(size: int = 15, output_dir: str = "map_previews"):
    """Generate preview images for all map types"""
    os.makedirs(output_dir, exist_ok=True)
    
    generator = MapGenerator()
    
    # All map types
    all_map_types = list(MapType)
    
    print(f"Generating {len(all_map_types)} map types at size {size}x{size}...")
    
    for map_type in all_map_types:
        try:
            # Generate map (connectivity is ensured automatically)
            grid_map = generator.generate(map_type, size=size)
            
            # Create figure
            _, ax = plt.subplots(figsize=(6, 6))
            ax.imshow(grid_map, cmap='gray_r', vmin=0, vmax=1)
            ax.set_title(f"{map_type.value}", fontsize=14, fontweight='bold')
            ax.axis('off')
            
            # Save
            filename = f"{map_type.value}.png"
            filepath = os.path.join(output_dir, filename)
            plt.savefig(filepath, dpi=150, bbox_inches='tight')
            plt.close()
            
            print(f"  ✓ {map_type.value}")
            
        except (ValueError, RuntimeError) as e:
            print(f"  ✗ {map_type.value}: {e}")
    
    print(f"\nAll previews saved to {output_dir}/")


if __name__ == "__main__":
    # Generate all map previews at fixed small size
    generate_all_previews(size=15)
