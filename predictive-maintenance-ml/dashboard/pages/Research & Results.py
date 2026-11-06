from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[2]
REPORT_DIR = PROJECT_ROOT / 'reports' / 'research_paper'
FD001_RESULTS_PATH = REPORT_DIR / 'fd001_raw_sensor_grouped_cv.json'
CROSS_SUBSET_RESULTS_PATH = REPORT_DIR / 'cross_subset_grouped_cv.json'
MANUSCRIPT_PATH = REPORT_DIR / 'manuscript.md'

MODEL_LABELS = {
    'random_forest': 'Random Forest',
    'ridge': 'Ridge',
    'linear_regression': 'Linear regression',
}
METRICS = {
    'RMSE (cycles)': ('rmse', 'Lower is better'),
    'MAE (cycles)': ('mae', 'Lower is better'),
    'R²': ('r2', 'Higher is better'),
    'NASA-style score (sum)': ('nasa_score', 'Lower is better'),
}
MODEL_COLORS = {
    'Random Forest': '#d27a3c',
    'Ridge': '#527b87',
    'Linear regression': '#8c927d',
}

st.set_page_config(
    page_title='Research & Results | C-MAPSS',
    page_icon='📊',
    layout='wide',
)

st.markdown(
    """
    <style>
    .stApp { background: #f5f3ed; }
    [data-testid="stHeader"] { background: rgba(245, 243, 237, 0.92); }
    [data-testid="stSidebar"] { background: #eae8df; }
    .block-container { max-width: 1440px; padding-top: 2rem; padding-bottom: 4rem; }
    .research-kicker {
        color: #b2602e; font-size: 0.76rem; font-weight: 750;
        letter-spacing: 0.16em; text-transform: uppercase; margin-bottom: 0.65rem;
    }
    .research-title {
        color: #202c2b; font-family: Georgia, 'Times New Roman', serif;
        font-size: clamp(2.6rem, 5vw, 4.7rem); line-height: 1.02;
        letter-spacing: -0.045em; margin: 0;
    }
    .research-deck {
        color: #596461; font-size: 1.12rem; line-height: 1.65;
        max-width: 820px; margin-top: 1rem;
    }
    .hero-meta {
        border-top: 1px solid #d5d4cb; border-bottom: 1px solid #d5d4cb;
        color: #65706b; font-size: 0.83rem; letter-spacing: 0.02em;
        margin: 1.6rem 0 2rem; padding: 0.9rem 0;
    }
    .section-label {
        color: #b2602e; font-size: 0.72rem; font-weight: 750;
        letter-spacing: 0.13em; text-transform: uppercase;
    }
    div[data-testid="stMetric"] {
        background: #fffefa; border: 1px solid #e3e0d7;
        border-radius: 12px; padding: 1rem 1.15rem;
    }
    div[data-testid="stMetricLabel"] { color: #65706b; }
    div[data-testid="stMetricValue"] { color: #202c2b; }
    .note-card {
        background: #eeece4; border-left: 3px solid #d27a3c;
        border-radius: 0 8px 8px 0; color: #4c5753;
        padding: 0.9rem 1.1rem; line-height: 1.55;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data(show_spinner=False)
def _read_json(path: Path) -> dict:
    with path.open(encoding='utf-8') as result_file:
        return json.load(result_file)


@st.cache_data(show_spinner=False)
def _read_text(path: Path) -> str:
    return path.read_text(encoding='utf-8')


def _load_required_file(path: Path, label: str) -> dict | str:
    if not path.is_file():
        st.error(f'{label} is missing: `{path}`')
        st.info('Restore the report file or regenerate it using the commands in the README.')
        st.stop()
    try:
        return _read_json(path) if path.suffix == '.json' else _read_text(path)
    except (OSError, json.JSONDecodeError, UnicodeDecodeError) as exc:
        st.error(f'Could not read {label}: {exc}')
        st.stop()


fd001_data = _load_required_file(FD001_RESULTS_PATH, 'FD001 benchmark results')
cross_data = _load_required_file(CROSS_SUBSET_RESULTS_PATH, 'Cross-subset benchmark results')
manuscript = _load_required_file(MANUSCRIPT_PATH, 'Research manuscript')

if not isinstance(fd001_data, dict) or not isinstance(cross_data, dict):
    st.error('Benchmark results must be valid JSON objects.')
    st.stop()
if not isinstance(manuscript, str):
    st.error('The research manuscript must be a readable text file.')
    st.stop()

subsets = cross_data.get('subsets')
if not isinstance(subsets, dict) or not subsets:
    st.error('Cross-subset results do not contain any dataset results.')
    st.stop()

st.sidebar.markdown('<div class="section-label">Study controls</div>', unsafe_allow_html=True)
subset_id = st.sidebar.selectbox('Dataset subset', list(subsets), index=0)
metric_label = st.sidebar.selectbox('Comparison metric', list(METRICS), index=0)
metric_key, metric_direction = METRICS[metric_label]
st.sidebar.caption(
    'Cross-subset metrics are evaluated independently within each subset. '
    'They are not cross-dataset transfer scores.'
)

subset_result = subsets[subset_id]
subset_models = subset_result.get('models', [])
if not subset_models or any('pooled_out_of_fold_metrics' not in model for model in subset_models):
    st.error(f'The benchmark data for {subset_id} is incomplete.')
    st.stop()

metric_records = []
for model_result in subset_models:
    model_name = MODEL_LABELS.get(model_result['model'], model_result['model'])
    metric_records.append(
        {
            'Model': model_name,
            metric_label: model_result['pooled_out_of_fold_metrics'][metric_key],
            'RMSE rank': model_result.get('rank'),
            'MAE (cycles)': model_result['pooled_out_of_fold_metrics']['mae'],
            'RMSE (cycles)': model_result['pooled_out_of_fold_metrics']['rmse'],
            'R²': model_result['pooled_out_of_fold_metrics']['r2'],
            'NASA-style score (sum)': model_result['pooled_out_of_fold_metrics']['nasa_score'],
        }
    )
metrics_frame = pd.DataFrame(metric_records)
best_index = (
    metrics_frame[metric_label].idxmax()
    if metric_key == 'r2'
    else metrics_frame[metric_label].idxmin()
)
best_model = metrics_frame.loc[best_index]

st.markdown('<div class="research-kicker">C-MAPSS · Predictive maintenance study</div>', unsafe_allow_html=True)
st.markdown('<h1 class="research-title">Research &amp; results</h1>', unsafe_allow_html=True)
st.markdown(
    '<p class="research-deck">An interactive report on engine-grouped remaining useful life '
    'estimation. Explore pooled out-of-fold results, compare model behavior across four '
    'C-MAPSS subsets, and read the complete reproducible study.</p>',
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="hero-meta">NASA C-MAPSS &nbsp; / &nbsp; 3-fold GroupKFold by engine '
    '&nbsp; / &nbsp; 3 conventional regression baselines &nbsp; / &nbsp; training data only</div>',
    unsafe_allow_html=True,
)

overview_tab, results_tab, methods_tab, paper_tab = st.tabs(
    ['Study overview', 'Explore results', 'Methods & interpretation', 'Full paper']
)

with overview_tab:
    st.markdown('<div class="section-label">Selected experiment</div>', unsafe_allow_html=True)
    st.subheader(f'{subset_id} · grouped validation')
    overview_columns = st.columns(4)
    overview_columns[0].metric('Training observations', f'{subset_result["rows"]:,}')
    overview_columns[1].metric('Engines', f'{subset_result["engines"]:,}')
    overview_columns[2].metric('Validation folds', f'{subset_result["validation"]["folds"]}')
    overview_columns[3].metric(
        f'Best {metric_label}',
        f'{best_model[metric_label]:,.3f}' if metric_key != 'nasa_score' else f'{best_model[metric_label]:,.0f}',
        str(best_model['Model']),
    )

    left, right = st.columns([1.3, 1])
    with left:
        st.markdown('<div class="section-label">Model comparison</div>', unsafe_allow_html=True)
        comparison = px.bar(
            metrics_frame.sort_values(metric_label, ascending=metric_key != 'r2'),
            x='Model',
            y=metric_label,
            color='Model',
            color_discrete_map=MODEL_COLORS,
            text=metrics_frame.sort_values(metric_label, ascending=metric_key != 'r2')[metric_label].map(
                lambda value: f'{value:,.3f}' if metric_key != 'nasa_score' else f'{value:,.0f}'
            ),
        )
        comparison.update_traces(textposition='outside', cliponaxis=False)
        comparison.update_layout(
            showlegend=False,
            height=390,
            margin=dict(l=12, r=12, t=25, b=12),
            yaxis_title=f'{metric_label} · {metric_direction.lower()}',
            xaxis_title=None,
            plot_bgcolor='#fffefa',
            paper_bgcolor='#fffefa',
            font=dict(color='#35413e'),
        )
        st.plotly_chart(comparison, use_container_width=True)
    with right:
        st.markdown('<div class="section-label">At a glance</div>', unsafe_allow_html=True)
        st.markdown(
            f"""
            Random Forest leads on pooled RMSE in **three of four subsets**. For **{subset_id}**,
            **{best_model['Model']}** ranks best by {metric_label.lower()} among the three
            evaluated baselines.

            This is an engine-grouped training-set comparison, not an official test-set score
            or evidence of real-world aircraft performance.
            """
        )
        st.markdown(
            '<div class="note-card">Each subset is trained and validated independently. '
            'Cross-subset performance does not measure transfer from one dataset to another.</div>',
            unsafe_allow_html=True,
        )

    st.markdown('<div class="section-label">Why engine grouping?</div>', unsafe_allow_html=True)
    st.markdown(
        'Rows from the same engine follow a correlated degradation trajectory. '
        'GroupKFold keeps each engine entirely within one fold, reducing the trajectory leakage '
        'that can happen when individual cycles are randomly split.'
    )

with results_tab:
    st.markdown('<div class="section-label">Interactive results explorer</div>', unsafe_allow_html=True)
    if metric_key == 'nasa_score':
        st.subheader(f'{metric_label} · {subset_id}')
    else:
        st.subheader(f'{metric_label} across independent subset runs')

    cross_records = []
    for dataset_name, result in subsets.items():
        for model_result in result.get('models', []):
            cross_records.append(
                {
                    'Subset': dataset_name,
                    'Model': MODEL_LABELS.get(model_result['model'], model_result['model']),
                    'RMSE (cycles)': model_result['pooled_out_of_fold_metrics']['rmse'],
                    'MAE (cycles)': model_result['pooled_out_of_fold_metrics']['mae'],
                    'R²': model_result['pooled_out_of_fold_metrics']['r2'],
                    'NASA-style score (sum)': model_result['pooled_out_of_fold_metrics']['nasa_score'],
                    'RMSE rank': model_result.get('rank'),
                }
            )
    cross_frame = pd.DataFrame(cross_records)
    chart_frame = (
        cross_frame.loc[cross_frame['Subset'] == subset_id]
        if metric_key == 'nasa_score'
        else cross_frame
    )
    subset_chart = px.bar(
        chart_frame,
        x='Subset',
        y=metric_label,
        color='Model',
        barmode='group',
        color_discrete_map=MODEL_COLORS,
        category_orders={'Subset': list(subsets)},
        hover_data=['RMSE rank'],
    )
    subset_chart.update_layout(
        height=440,
        margin=dict(l=12, r=12, t=20, b=12),
        yaxis_title=f'{metric_label} · {metric_direction.lower()}',
        plot_bgcolor='#fffefa',
        paper_bgcolor='#fffefa',
        font=dict(color='#35413e'),
        legend_title=None,
    )
    st.plotly_chart(subset_chart, use_container_width=True)
    if metric_key == 'nasa_score':
        st.caption(
            'The NASA-style score is a sum whose magnitude depends on the number of evaluated '
            'observations. It is restricted to this selected subset and should not be compared '
            'across datasets.'
        )
    else:
        st.caption(
            'Subset scores are evaluated independently. Differences in operating conditions, '
            'failure modes, and sample counts mean raw values across subsets are not transfer comparisons.'
        )

    rank_column, detail_column = st.columns([0.85, 1.3])
    with rank_column:
        st.markdown('<div class="section-label">RMSE rank consistency</div>', unsafe_allow_html=True)
        rank_frame = pd.DataFrame(cross_data.get('rank_consistency', []))
        if rank_frame.empty:
            st.warning('Rank-consistency results are not available in the benchmark file.')
        else:
            rank_frame['Model'] = rank_frame['model'].map(
                lambda model: MODEL_LABELS.get(model, model)
            )
            rank_figure = px.bar(
                rank_frame.sort_values('mean_rank'),
                x='Model',
                y='mean_rank',
                color='Model',
                color_discrete_map=MODEL_COLORS,
                text='mean_rank',
                hover_data={'rank_1_count': True, 'subsets_evaluated': True},
            )
            rank_figure.update_traces(texttemplate='%{text:.2f}', textposition='outside')
            rank_figure.update_layout(
                showlegend=False,
                height=350,
                margin=dict(l=12, r=12, t=20, b=12),
                yaxis_title='Mean rank (1 = best)',
                xaxis_title=None,
                plot_bgcolor='#fffefa',
                paper_bgcolor='#fffefa',
                font=dict(color='#35413e'),
            )
            st.plotly_chart(rank_figure, use_container_width=True)
    with detail_column:
        st.markdown(f'<div class="section-label">{subset_id} · pooled metrics</div>', unsafe_allow_html=True)
        display_columns = [
            'Model',
            'RMSE rank',
            'MAE (cycles)',
            'RMSE (cycles)',
            'R²',
            'NASA-style score (sum)',
        ]
        st.dataframe(
            metrics_frame[display_columns].style.format(
                {
                    'MAE (cycles)': '{:.3f}',
                    'RMSE (cycles)': '{:.3f}',
                    'R²': '{:.3f}',
                    'NASA-style score (sum)': '{:,.0f}',
                }
            ),
            hide_index=True,
            use_container_width=True,
        )

    fd001_models = fd001_data.get('models', [])
    fold_records = []
    for model_result in fd001_models:
        for fold_index, fold_metrics in enumerate(model_result.get('fold_metrics', []), start=1):
            fold_records.append(
                {
                    'Model': MODEL_LABELS.get(model_result['model'], model_result['model']),
                    'Fold': f'Fold {fold_index}',
                    'RMSE (cycles)': fold_metrics['rmse'],
                }
            )
    if fold_records:
        st.markdown('<div class="section-label">FD001 fold-level variation</div>', unsafe_allow_html=True)
        fold_figure = px.line(
            pd.DataFrame(fold_records),
            x='Fold',
            y='RMSE (cycles)',
            color='Model',
            markers=True,
            color_discrete_map=MODEL_COLORS,
        )
        fold_figure.update_layout(
            height=360,
            margin=dict(l=12, r=12, t=15, b=12),
            plot_bgcolor='#fffefa',
            paper_bgcolor='#fffefa',
            font=dict(color='#35413e'),
            legend_title=None,
        )
        st.plotly_chart(fold_figure, use_container_width=True)
        st.caption(
            'Fold dispersion is descriptive (three folds), not a confidence interval or significance test.'
        )

    st.download_button(
        'Download cross-subset results (JSON)',
        data=json.dumps(cross_data, indent=2).encode('utf-8'),
        file_name='cross_subset_grouped_cv.json',
        mime='application/json',
    )

with methods_tab:
    st.markdown('<div class="section-label">Protocol & responsible interpretation</div>', unsafe_allow_html=True)
    method_left, method_right = st.columns(2)
    with method_left:
        st.subheader('Data and target')
        st.markdown(
            """
            The study uses NASA C-MAPSS training observations. Each cycle has three operating
            settings and 21 sensor measurements; engine identifier and cycle are not model inputs.
            The uncapped target is computed per engine:

            **RUL(engine, cycle) = maximum observed training cycle − current cycle**

            The FD001 run contains 20,631 observations from 100 engines. The cross-subset runs
            repeat the same protocol independently on FD001–FD004.
            """
        )
        st.subheader('Compared baselines')
        st.markdown(
            '- **Linear regression:** intercept enabled.\n'
            '- **Ridge:** project default regularization, α = 1.0.\n'
            '- **Random Forest:** 300 trees, unrestricted depth, minimum leaf size 2, seed 42.\n\n'
            'All use the same 24 raw setting/sensor inputs; this baseline includes no tuning.'
        )
    with method_right:
        st.subheader('Validation')
        st.markdown(
            'Three-fold **GroupKFold by engine** assigns every engine to one fold. Every cycle '
            'receives an out-of-fold prediction. Pooled metrics concatenate those predictions '
            'and weight cycles equally, so engines with longer observed trajectories contribute '
            'more rows.'
        )
        st.subheader('Metrics')
        st.markdown(
            '- **MAE / RMSE:** cycle-level absolute and squared-error summaries.\n'
            '- **R²:** coefficient of determination on pooled out-of-fold predictions.\n'
            '- **NASA-style score:** asymmetric exponential penalty summed across observations; '
            'lower is better and the total depends on the number of evaluated observations.'
        )
        st.latex(
            r"s(d)=\begin{cases} e^{-d/13}-1,&d<0\\ e^{d/10}-1,&d\geq 0\end{cases},"
            r"\quad d=\widehat{\mathrm{RUL}}-\mathrm{RUL}"
        )

    st.subheader('Limitations')
    st.markdown(
        """
        - Three folds provide descriptive variability only; no repeated validation, confidence
          interval, or significance test was performed.
        - The reported baseline uses uncapped RUL and current-cycle measurements only.
        - No trained model artifact or official held-out test-set evaluation is reported.
        - Cross-subset runs do not train on one subset and test on another.
        - C-MAPSS is simulated data; results do not establish real aircraft performance.
        - Dashboard anomaly flags are exploratory screening signals, not diagnoses or RUL predictions.
        """
    )

with paper_tab:
    st.markdown('<div class="section-label">Research manuscript</div>', unsafe_allow_html=True)
    st.download_button(
        'Download manuscript (Markdown)',
        data=manuscript.encode('utf-8'),
        file_name='manuscript.md',
        mime='text/markdown',
    )
    st.markdown(manuscript)
