# Baadal: 3-minute pitch and demo script

Target is about 400 spoken words. Times are cumulative. Screen-record the application (`streamlit run app.py`) in a full-width browser window. The progress tracker at the top of the app follows the same five steps as this script.

---

**0:00-0:25 | Problem (title slide or camera)**
> In 2010 and again in 2022, floods put a large part of Pakistan under water. Every monsoon, flood managers get a forecast a few days ahead and must answer three questions fast: how serious is it, how likely is it, and what do we do with a limited budget? Baadal, which means "cloud", answers all three.

**0:25-0:45 | What Baadal does (architecture slide, `docs/figures/architecture.png`)**
> Baadal collects real rain forecasts, forecast ensembles and river flow for twelve Pakistani cities, and models the flood on each city's real elevation and land cover. Qiskit does two jobs: quantum amplitude estimation estimates the flood risk, and QAOA chooses the response plan. Baadal does not predict the weather; it acts on the forecast.

**0:45-1:00 | Step 1, choose city (app, "Live forecast", then "Replay a past flood")**
> This is live: today's forecast shows no flood threat, which is right for November. So let's replay the 2010 super flood from recorded data. Seven of twelve cities are under threat. I pick Sukkur.

**1:00-1:25 | Step 2, assess threat**
> The map flies to Sukkur on real satellite imagery. The Indus peaks at 1.8 times the flow at which flooding starts. The chance of flooding is estimated on four qubits with amplitude estimation, and it matches the exact answer.

**1:25-1:45 | Step 3, see impact (click "See the impact")**
> If nothing is done, this is where the water goes. The flood follows the real ground: the low land along the river floods, and the high ridge stays dry. Baadal totals the damage and the people exposed, area by area.

**1:45-2:25 | Step 4 and 5, plan and review (click "Plan the response", then "Review plan")**
> Now the response. This works like placing an order: a menu of protective measures, each with a cost, and a budget. Each measure is one qubit. Their interactions, like a breach needing evacuation first, make this a QUBO, and QAOA picks the best set. Qiskit's picks are marked, and the map shows the result. I can also build my own plan: when I add stone spurs, the summary tells me I am over budget. On review, Baadal confirms that the quantum plan is the best of all 1,024 possible plans.

**2:25-2:50 | Results and limits (open "Technical details", Limitations tab)**
> In simulation, our quantum risk estimate needed about four times fewer samples than classical sampling, as the theory predicts. Now the limits. These are small problems on a simulator, so classical methods can also solve them exactly, and that is how we check every quantum answer. QAOA does not always find the best plan, and the risk circuits are too deep for today's noisy hardware. The flood map is a simple fill model, and the damage and cost figures are illustrative.

**2:50-3:00 | Close**
> What we built is a complete, tested application, from live weather to a reviewed response plan, with quantum steps that do real work and are checked against exact answers. Baadal.

---

**Recording tips**
- Use a browser window at least 1,400 pixels wide so the map and the panel sit side by side.
- Click through the demo once before recording. Plans for the saved floods at the default budget are pre-computed, so they appear at once. A new budget or city takes 20 to 30 seconds the first time.
- The satellite imagery needs an internet connection.
- Check the Sukkur numbers on your screen before you say them (1.8 times, seven of twelve).
- Hard stop at 3:00. The rules penalise longer videos.
