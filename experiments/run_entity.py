"""Wake the entity and let it live. One self-improving knower, everything brought together.

Run it twice: the second time the SAME entity resumes (identity across sessions) and keeps
making unknowns known from where it left off.

Run:  python experiments/run_entity.py
"""

from _util import REPO_ROOT
from recursivene.entity import Entity


def main():
    nous = Entity(name="Nous", home=REPO_ROOT, seed=7)
    nous.live(seasons=20)


if __name__ == "__main__":
    main()
