# Baadal: 3-minute pitch and demo script

About 420 spoken words. Times are cumulative. The framing is deliberate: say clearly what quantum does here and what it does not, before the judges ask. No claim in this script goes beyond the results in the repository.

Slides needed: (1) title, (2) `docs/figures/architecture.png`, (3) `docs/figures/fig_qae_benchmark.png`. Everything else is the live app (`streamlit run app.py`).

---

**0:00-0:20 | Problem (title slide)**
> In 2010 and 2022, floods put a large part of Pakistan under water. When a forecast arrives, flood managers have days to decide: how serious is it, how likely is it, and what should we fund with a limited budget? We built Baadal, which means "cloud", to answer those three questions in one tool.

**0:20-0:45 | What it is, and the honest framing (architecture slide)**
> Baadal collects real weather and river forecasts for twelve Pakistani cities, detects threats, models the flood on real satellite terrain, and recommends a response. Two steps run on Qiskit: amplitude estimation for the flood risk, and QAOA for the response plan.
> We want to be upfront: at the size we run today, a classical computer can solve both steps exactly. We use that to check every quantum answer. What we built is the quantum formulation of a real decision, validated now, and designed for the scale where checking every plan stops being practical.

**0:45-1:05 | Step 1, choose city (app: Live forecast, then Replay a past flood)**
> This is today's live forecast: no flood threat, which is right for November. So let's replay the 2010 super flood from recorded data. Seven of twelve cities are flagged. We pick Sukkur.

**1:05-1:25 | Step 2, assess threat**
> The Indus at Sukkur peaked at 1.8 times the flow where flooding starts. The flood chance comes from quantum amplitude estimation over the forecast scenarios, and it matches the exact answer.

**1:25-1:40 | Step 3, see impact (click See the impact, pause while the water spreads)**
> If nothing is done, this is where the water goes, following the real ground. The panel totals damage and people at risk by area. The cost figures are illustrative.

**1:40-2:15 | Steps 4 and 5, plan and review**
> Now the response. Each protective measure is one qubit. Their interactions, like a breach that is only allowed after evacuation, become a QUBO, and QAOA picks the plan within budget. Here it is the best of all 1,024 possible plans, which we confirm classically. Planners can also build their own plan: add stone spurs, and Baadal warns that it is over budget.

**2:15-2:45 | Results and limits (results slide: QAE graph)**
> On an ideal simulator, our quantum risk estimate needed about four and a half times fewer samples than classical sampling for the same accuracy, the speed-up theory predicts. With realistic noise, that gain disappears: this needs future error-corrected hardware. QAOA matched the best plan in all nine real-event cases, but in our benchmark a simple classical search did nearly as well. At ten measures, quantum is not needed. A district has forty or more, about a trillion possible plans, and that is where this formulation is aimed.

**2:45-3:00 | Close (back to the map)**
> Baadal is a working application, from live weather to a verified plan, with the quantum steps built, tested, and honest about where they stand. Thank you.

---

## Q&A preparation

Each team member should be able to give these answers in their own words.

**"Why use quantum if a laptop solves it?"**
> At this size it isn't needed, and we say so. We use the small size to verify the quantum steps against exact answers. The value is a validated formulation that scales: the number of plans doubles with every measure added, so a district-scale problem cannot be checked plan by plan.

**"Does QAOA beat classical methods?"**
> No. It found the best plan in 10 of 12 benchmark cases; random sampling plus local search found it in 9. Whether QAOA helps at large scale is an open research question.

**"What exactly does amplitude estimation compute?"**
> The probability that rain or river flow crosses the flood level, from the forecast ensemble. We load the distribution into 3 qubits, flag the flooding outcomes on a fourth, amplify that flag with Grover-style rotations, and fit the probability from the measurements.

**"Why does the noisy line in the graph go flat?"**
> The more accurate runs use deeper circuits, with up to 673 two-qubit gates. Each gate adds a little error, and the errors swamp the gain. That's why this needs error-corrected hardware.

**"Are the damage numbers real?"**
> No. Weather, river flow, elevation, land cover and imagery are real public data. Asset values, population density and measure costs are illustrative.

**"Did you use AI?"**
> Yes, as a coding assistant, and the README says so. We reviewed, ran and tested the code, and we can explain each part.

## Recording tips

- Start the app 5 minutes early and click through the demo path once.
- Keep the internet on, and don't press "Update weather" while recording.
- Pause after each click so the animation finishes before you speak.
- Read the script once with a timer. Cut words from the intro if you run over 3:00.
