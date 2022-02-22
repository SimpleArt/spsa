import examples.ackley as ackley
import examples.beale as beale
import examples.ellipse as ellipse
import examples.rastrigin as rastrigin
import examples.rosenbrock as rosenbrock
import examples.sphere as sphere

__all__ = ["ackley", "beale", "ellipse", "main", "rastrigin", "rosenbrock", "sphere"]

def main() -> None:
    print()
    print("All examples:")
    print("-" * 140)
    ackley.main()
    ellipse.main()
    rastrigin.main()
    rosenbrock.main()
    sphere.main()

if __name__ == "__main__":
    main()
