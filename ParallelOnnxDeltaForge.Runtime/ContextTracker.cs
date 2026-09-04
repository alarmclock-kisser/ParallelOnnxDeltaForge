using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Linq;
using System.Threading;
using ParallelOnnxDeltaForge.Shared;
using ParallelOnnxDeltaForge.Shared.Dtos;

namespace ParallelOnnxDeltaForge.Runtime
{
    public class ContextTracker : ParallelOnnxDeltaForge.Shared.Interfaces.IContextTracker
    {
        private readonly ConcurrentBag<ContextTurn> _turns = new();
        private int _nextIndex;

        public int TurnCount => this._turns.Count;

        public void RecordTurn(ContextTurn turn)
        {
            turn.TurnIndex = Interlocked.Increment(ref this._nextIndex) - 1;
            turn.Timestamp = DateTime.UtcNow;
            this._turns.Add(turn);
        }

        public IReadOnlyList<ContextTurn> GetTurns()
        {
            var list = this._turns.ToList();
            list.Sort((a, b) => a.TurnIndex.CompareTo(b.TurnIndex));
            return list.AsReadOnly();
        }

        public void Clear()
        {
            this._turns.Clear();
            Interlocked.Exchange(ref this._nextIndex, 0);
        }
    }
}
