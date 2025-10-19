package net.tvburger.jdl.model.training.regularization;

/**
 * Explicit regularization involves modifying the objective (loss) function directly by adding a penalty or changing the targets.
 *
 * @param <N> Number type
 */
@Regularization.Mechanism(Regularization.Mechanism.Type.PENALTY_BASED)
public interface ExplicitRegularization<N extends Number> extends Regularization<N> {

    N lossPenalty(N[] parameters);

    N gradientAdjustment(N parameter);

}
