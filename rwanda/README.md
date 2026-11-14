# The Drying Floor at Karongi (Rwanda)

**Flag: `SIGMOID_CONVERGENCE`**

## The bug and the fix
In the handed-out LangGraph state machine, `inspect_batch` measured the lot's moisture and computed whether it was inside the
accepted range, but never wrote that decision back into the state. It returned only the inspection counter. So `approved` stayed
`False`, the router kept sending every batch back to `dry_batch`, and every batch ran out of the step budget.

The fix is one line in `inspect_batch`, which now also returns the decision:
```python
return {"inspections": state["inspections"] + 1, "approved": ready}
```
Nothing else changed. Constants, state, the other nodes, the router and the graph are identical to the reference structure,
which is what the organizers' structural check requires.

## Result (`run_log.txt`)
All 8 batches converge inside the step budget, and each gets a grade. The graph prints `SIGMOID_CONVERGENCE`.

## Run
Put `graph.py` together with the task's `dataset/batches.json` and `evaluation_script.py`.
```
pip install langgraph
python rwanda/graph.py
```
