from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.ensemble import IsolationForest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.dashboard.data import discover_available_dataset_ids, load_dashboard_dataset

RAW_DATA_DIR = PROJECT_ROOT / 'data' / 'raw' / 'CMAPSSData'


@st.cache_data(show_spinner=False)
def _load_dataset(fd_id: str) -> dict[str, pd.DataFrame]:
    return load_dashboard_dataset(fd_id, RAW_DATA_DIR)


st.set_page_config(
    page_title='C-MAPSS Predictive Maintenance',
    page_icon='✈️',
    layout='wide',
)
st.title('C-MAPSS Engine Health Dashboard')
st.caption(
    'Explore NASA C-MAPSS engine trajectories, reference RUL labels, and unsupervised '
    'sensor outlier signals. Outlier flags are screening indicators, not fault diagnoses.'
)

available_ids = discover_available_dataset_ids(RAW_DATA_DIR)
if not available_ids:
    st.error(f'No complete C-MAPSS dataset is available in `{RAW_DATA_DIR}`.')
    st.info('Place train_FDxxx.txt, test_FDxxx.txt, and RUL_FDxxx.txt together in that folder.')
    st.stop()

with st.sidebar:
    st.header('Dataset')
    fd_id = st.selectbox('C-MAPSS subset', available_ids)
    split = st.radio('Data split', ['train', 'test'], horizontal=True)

try:
    data = _load_dataset(fd_id)
except (FileNotFoundError, ValueError, OSError, pd.errors.ParserError) as exc:
    st.error(f'Could not load {fd_id}: {exc}')
    st.stop()

frame = data[split]
sensor_columns = [column for column in frame.columns if column.startswith('sensor_')]
engine_ids = sorted(frame['engine_id'].unique().tolist())

with st.sidebar:
    st.header('Trajectory')
    engine_id = st.selectbox('Engine', engine_ids)
    sensor = st.selectbox('Sensor', sensor_columns, index=min(1, len(sensor_columns) - 1))

engine_data = frame.loc[frame['engine_id'] == engine_id].sort_values('cycle')

metric_columns = st.columns(4)
metric_columns[0].metric('Engines', f'{frame["engine_id"].nunique():,}')
metric_columns[1].metric('Observations', f'{len(frame):,}')
metric_columns[2].metric('Maximum observed cycle', f'{int(frame["cycle"].max()):,}')
metric_columns[3].metric('Mean reference RUL', f'{frame["rul"].mean():.1f} cycles')

trajectory_tab, anomaly_tab, data_tab = st.tabs(
    ['Engine trajectory', 'Anomaly screening', 'Dataset preview']
)

with trajectory_tab:
    left, right = st.columns(2)
    sensor_figure = px.line(
        engine_data,
        x='cycle',
        y=sensor,
        markers=True,
        title=f'Engine {engine_id}: {sensor}',
        labels={'cycle': 'Operating cycle', sensor: sensor},
    )
    left.plotly_chart(sensor_figure, use_container_width=True)

    rul_figure = px.line(
        engine_data,
        x='cycle',
        y='rul',
        markers=True,
        title='Reference RUL over observed cycles',
        labels={'cycle': 'Operating cycle', 'rul': 'Reference RUL (cycles)'},
    )
    right.plotly_chart(rul_figure, use_container_width=True)
    if split == 'test':
        st.caption(
            'Test-set RUL is reconstructed from NASA ground truth at each engine’s final '
            'observed cycle; these are reference labels, not model predictions.'
        )
    else:
        st.caption('Training RUL labels are max observed engine cycle minus current cycle.')

with anomaly_tab:
    st.write(
        'Fit an Isolation Forest on the selected subset’s training sensor/settings and '
        'score the chosen split. This is unsupervised outlier screening.'
    )
    contamination = st.slider(
        'Expected outlier fraction',
        min_value=0.01,
        max_value=0.20,
        value=0.05,
        step=0.01,
    )
    if st.button('Run anomaly screening', type='primary'):
        train = data['train']
        feature_columns = [
            column for column in train.columns
            if (column.startswith('sensor_') or column.startswith('setting_'))
            and pd.api.types.is_numeric_dtype(train[column])
        ]
        model = IsolationForest(
            contamination=contamination,
            random_state=42,
            n_jobs=-1,
        )
        model.fit(train[feature_columns])
        scored = frame[['engine_id', 'cycle']].copy()
        scored['anomaly_score'] = -model.score_samples(frame[feature_columns])
        scored['is_anomaly'] = model.predict(frame[feature_columns]) == -1
        anomalies = int(scored['is_anomaly'].sum())
        st.metric('Flagged observations', f'{anomalies:,}', f'{anomalies / len(scored):.1%} of split')

        selected_engine_scores = scored.loc[scored['engine_id'] == engine_id]
        anomaly_figure = px.scatter(
            selected_engine_scores,
            x='cycle',
            y='anomaly_score',
            color='is_anomaly',
            title=f'Engine {engine_id}: outlier score by cycle',
            labels={'cycle': 'Operating cycle', 'anomaly_score': 'Outlier score'},
            color_discrete_map={False: '#1f77b4', True: '#d62728'},
        )
        st.plotly_chart(anomaly_figure, use_container_width=True)
        st.dataframe(
            scored.loc[scored['is_anomaly']].sort_values('anomaly_score', ascending=False).head(100),
            use_container_width=True,
        )

with data_tab:
    st.dataframe(frame.head(1000), use_container_width=True)
    st.download_button(
        'Download selected split as CSV',
        data=frame.to_csv(index=False).encode('utf-8'),
        file_name=f'{fd_id}_{split}.csv',
        mime='text/csv',
    )
