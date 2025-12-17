from evolve import random_graph, mutate, fitness
import random

POP_SIZE = 20
GENERATIONS = 50
OFFSPRING = 20

# -------------------------
# Init population
# -------------------------

population = [random_graph() for _ in range(POP_SIZE)]


# -------------------------
# GA loop
# -------------------------

for gen in range(GENERATIONS):
    scored = [(fitness(g), g) for g in population]
    scored.sort(key=lambda x: x[0], reverse=True)

    best = scored[0][0]
    diversity = len({g.hash() for _, g in scored})
    nodes = len(scored[0][1].nodes)

    print(
        f"gen={gen:03d} "
        f"best={best:7.3f} "
        f"diversity={diversity:02d} "
        f"nodes={nodes}"
    )

    # offspring
    children = []
    success = 0

    for _ in range(OFFSPRING):
        parent = random.choice(scored[:5])[1]
        child = mutate(parent)
        if fitness(child) > fitness(parent):
            success += 1
        children.append(child)

    # (μ + λ) selection
    population = [
        g for _, g in sorted(
            [(fitness(g), g) for g in population + children],
            key=lambda x: x[0],
            reverse=True
        )[:POP_SIZE]
    ]
