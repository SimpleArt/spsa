import examples.ackley as ackley
import examples.ellipse as ellipse
import examples.ellipse_with_noise as ellipse_with_noise
import examples.rastrigin as rastrigin
import examples.rosenbrock as rosenbrock
import examples.sphere as sphere
import examples.sphere_with_noise as sphere_with_noise

__all__ = ["ackley", "ellipse", "ellipse_with_noise", "main", "rastrigin", "rosenbrock", "sphere", "sphere_with_noise"]

def main() -> None:
    print()
    print("All examples:")
    print("-" * 140)
    ackley.main()
    ellipse.main()
    ellipse_with_noise.main()
    rastrigin.main()
    rosenbrock.main()
    sphere.main()
    sphere_with_noise.main()

if __name__ == "__main__":
    main()
