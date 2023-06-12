import math
from itertools import product
from typing import Optional, Set, Sequence, List

from posggym.envs.grid_world.core import Grid, Coord


class PPGrid(Grid):
    """A grid for the Predator-Prey Problem."""

    def __init__(
        self,
        grid_size: int,
        block_coords: Optional[Set[Coord]],
        predator_start_coords: Optional[List[Coord]] = None,
        prey_start_coords: Optional[List[Coord]] = None,
    ):
        assert grid_size >= 3
        super().__init__(grid_size, grid_size, block_coords)
        self.size = grid_size
        # predators start in corners or half-way along a side
        if predator_start_coords is None:
            predator_start_coords = list(
                c
                for c in product([0, grid_size // 2, grid_size - 1], repeat=2)
                if c[0] in (0, grid_size - 1) or c[1] in (0, grid_size - 1)
            )
        self.predator_start_coords = predator_start_coords
        self.prey_start_coords = prey_start_coords

    def get_ascii_repr(
        self,
        predator_coords: Optional[Sequence[Coord]],
        prey_coords: Optional[Sequence[Coord]],
    ) -> str:
        """Get ascii repr of grid."""
        grid_repr = []
        for row in range(self.height):
            row_repr = []
            for col in range(self.width):
                coord = (col, row)
                if coord in self.block_coords:
                    row_repr.append("#")
                else:
                    row_repr.append(".")
            grid_repr.append(row_repr)

        if predator_coords is not None:
            for c in predator_coords:
                grid_repr[c[0]][c[1]] = "P"
        if prey_coords is not None:
            for c in predator_coords:
                grid_repr[c[0]][c[1]] = "p"

        return (
            str(self) + "\n" + "\n".join(list(list((" ".join(r) for r in grid_repr))))
        )

    def get_unblocked_center_coords(self, num: int) -> List[Coord]:
        """Get at least num closest coords to the center of grid.

        May return more than num, since can be more than one coord at equal
        distance from the center.
        """
        assert num < self.n_coords - len(self.block_coords)
        center = (self.width // 2, self.height // 2)
        min_dist_from_center = math.ceil(math.sqrt(num)) - 1
        coords = self.get_coords_within_dist(
            center, min_dist_from_center, ignore_blocks=False, include_origin=True
        )

        while len(coords) < num:
            # not the most efficient as it repeats work,
            # but function should only be called once when model is initialized
            # and for small num
            for c in coords:
                coords.update(
                    self.get_neighbours(
                        c, ignore_blocks=False, include_out_of_bounds=False
                    )
                )

        return list(coords)

    def num_unblocked_neighbours(self, coord: Coord) -> int:
        """Get number of neighbouring coords that are unblocked."""
        return len(
            self.get_neighbours(coord, ignore_blocks=False, include_out_of_bounds=False)
        )


def get_10x10_grid() -> PPGrid:
    """Generate 10x10 grid layout."""
    return PPGrid(grid_size=10, block_coords=None)


#  (grid_make_fn, step_limit)
SUPPORTED_GRIDS = {
    "10x10": (get_10x10_grid, 50),
}


def load_grid(grid_name: str) -> PPGrid:
    """Load grid with given name."""
    grid_name = grid_name.lower()
    assert grid_name in SUPPORTED_GRIDS, (
        f"Unsupported grid name '{grid_name}'. Grid name must be one of: "
        f"{SUPPORTED_GRIDS.keys()}."
    )
    return SUPPORTED_GRIDS[grid_name][0]()
