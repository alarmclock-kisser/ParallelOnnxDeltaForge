using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using ParallelOnnxDeltaForge.Shared.Dtos;

namespace ParallelOnnxDeltaForge.Runtime
{
    /// <summary>
    /// Computes LoRA weight deltas from accumulated chat context.
    /// Uses thin SVD on the delta-output × input^T matrix to produce rank-k decomposition.
    /// </summary>
    public class LoRADeltaComputationService : ParallelOnnxDeltaForge.Shared.Interfaces.IDeltaComputationService
    {
        private const float Epsilon = 1e-8f;

        public async Task<LoRADeltaSet> ComputeFromContextAsync(IReadOnlyList<ContextTurn> turns, int targetRank)
        {
            if (turns.Count == 0)
                throw new ArgumentException("No context turns available for delta computation.", nameof(turns));

            // Guard: zero rank → return empty set with correct rank
            if (targetRank == 0)
                return new LoRADeltaSet { Name = $"delta_set_{turns.Count}_turns", Rank = 0, AccumulatedTurns = turns.Count };

            // Guard: negative rank → clamp to 1
            if (targetRank < 0)
                targetRank = 1;

            // Guard: rank must be >= 1
            if (targetRank <= 0)
                targetRank = 1;

            // Accumulate aggregate delta vectors: sum of (loraOutput - baseOutput) across turns
            float[]? aggregateDelta = null;
            float[]? aggregateInput = null;

            foreach (var turn in turns)
            {
                float[]? delta = this.ComputeTurnDelta(turn);
                if (delta == null || delta.Length == 0) continue;

                if (aggregateDelta == null)
                {
                    aggregateDelta = new float[delta.Length];
                    aggregateInput = turn.InputData != null ? new float[turn.InputData.Length] : new float[delta.Length];
                }

                for (int i = 0; i < delta.Length && i < aggregateDelta.Length; i++)
                    aggregateDelta[i] += delta[i];

                if (turn.InputData != null)
                {
                    for (int i = 0; i < turn.InputData!.Length && i < aggregateInput!.Length; i++)
                        aggregateInput![i] += turn.InputData[i];
                }
            }

            if (aggregateDelta == null || aggregateInput == null)
            {
                return new LoRADeltaSet { Name = "empty_delta", Rank = targetRank, AccumulatedTurns = 0 };
            }

            // Thin SVD: [dOut] × [dIn]^T → U[k, dOut] × S[k] × Vt[k, dIn]
            (var U, var S, var Vt) = this.ThinSVD(aggregateDelta, aggregateInput, targetRank);

            // Build LoRA deltas: A = S^(1/2) × Vt  [rank × dIn], B = U × S^(1/2)  [dOut × rank]
            var deltas = new Dictionary<string, LoRADelta>();

            float[] AData = new float[targetRank * aggregateInput.Length];
            float[] BData = new float[aggregateDelta.Length * targetRank];

            for (int k = 0; k < targetRank && k < S.Length; k++)
            {
                float sqrtS = (float)Math.Sqrt(S[k] + Epsilon);

                // A[k, :] = sqrtS * Vt[k, :]
                for (int j = 0; j < aggregateInput.Length && k + j * targetRank < Vt.Length; j++)
                    AData[k * aggregateInput.Length + j] = Vt[k + j * targetRank] * sqrtS;

                // B[:, k] = sqrtS * U[:, k]  →  B[k * dOut + i]
                for (int i = 0; i < aggregateDelta.Length && k + i * targetRank < U.Length; i++)
                    BData[k * aggregateDelta.Length + i] = U[k + i * targetRank] * sqrtS;
            }

            deltas["output_layer"] = new LoRADelta
            {
                LayerName = "output_layer",
                AShape = new long[] { targetRank, aggregateInput.Length },
                AData = AData,
                BShape = new long[] { aggregateDelta.Length, targetRank },
                BData = BData,
                ScaleFactor = S.Length > 0 ? (float)S[0] : 1f,
            };

            return new LoRADeltaSet
            {
                Name = $"delta_set_{turns.Count}_turns",
                Rank = targetRank,
                Deltas = deltas,
                AccumulatedTurns = turns.Count,
            };
        }

        private float[]? ComputeTurnDelta(ContextTurn turn)
        {
            if (turn.BaseOutputData == null || turn.LoraOutputData == null) return null;
            if (turn.BaseOutputData.Length != turn.LoraOutputData.Length) return null;

            var delta = new float[turn.BaseOutputData.Length];
            for (int i = 0; i < delta.Length; i++)
                delta[i] = turn.LoraOutputData[i] - turn.BaseOutputData[i];

            return delta;
        }

        /// <summary>
        /// Computes thin SVD of the rank-1 matrix M = y × x^T, where y = delta vector, x = input vector.
        /// Returns U[m×k], S[k], Vt[k×n]. For rank-1 input, only S[0] is non-zero.
        /// </summary>
        private (float[] U, double[] S, float[] Vt) ThinSVD(float[] y, float[] x, int k)
        {
            int m = y.Length;
            int n = x.Length;

            var U = new float[m * k];
            var Vt = new float[k * n];
            var S = new double[k];

            float yNorm = 0f, xNorm = 0f;
            for (int i = 0; i < y.Length; i++) yNorm += y[i] * y[i];
            for (int i = 0; i < x.Length; i++) xNorm += x[i] * x[i];

            yNorm = (float)Math.Sqrt(yNorm + Epsilon);
            xNorm = (float)Math.Sqrt(xNorm + Epsilon);

            double sigma = yNorm * xNorm;
            S[0] = sigma;

            for (int i = 0; i < m; i++) U[i] = y[i] / yNorm;
            for (int i = 0; i < n; i++) Vt[i] = x[i] / xNorm;

            for (int r = 1; r < k; r++) S[r] = 0;

            return (U, S, Vt);
        }
    }
}
