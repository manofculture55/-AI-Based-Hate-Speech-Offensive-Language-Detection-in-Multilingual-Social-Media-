from nicegui import ui, app, events
import numpy as np
import os
import time
import sqlite3
import pandas as pd
import json
import asyncio
from tensorflow.keras.models import load_model
import io

# --- IMPORT PROJECT MODULES ---
from src.models.bilstm import DeepModel
from src.data.preprocess import clean_text
from src.data.langid import detect_language_strict
from src.training.train import run_training

# --- 1. GLOBAL STATE & MODEL LOADING ---
print("  [App] Initializing KRIXION AI Engine...")
model_wrapper = DeepModel(architecture='bilstm')

try:
    model_wrapper.load_tokenizer()
    model_path = "models/deep/bilstm_model.h5"
    if os.path.exists(model_path):
        model_wrapper.model = load_model(model_path)
        print("  [App] BiLSTM Model Loaded Successfully!")
    else:
        print(f"  [App] Error: Model not found at {model_path}")
except Exception as e:
    print(f"  [App] Critical Error Loading Model: {e}")

# chart state
chart_data = {'home_pie': [], 'analytics_bar': []}

# --- 2. DATABASE HELPERS ---
def get_db_connection():
    return sqlite3.connect('data/app.db')

def fetch_chart_data():
    """Updates global chart data from DB"""
    try:
        conn = get_db_connection()

        # Home: last 10 predictions
        df_recent = pd.read_sql_query(
            "SELECT predicted_label FROM predictions ORDER BY id DESC LIMIT 10",
            conn
        )
        # counts_recent is a Series; convert to dict of class->count
        counts_recent = (
            df_recent['predicted_label']
            .value_counts()
            .reindex([0, 1, 2], fill_value=0)
            .to_dict()
        )

        # Analytics: all time
        df_all = pd.read_sql_query(
            "SELECT predicted_label FROM predictions",
            conn
        )
        counts_all = (
            df_all['predicted_label']
            .value_counts()
            .reindex([0, 1, 2], fill_value=0)
            .to_dict()
        )

        conn.close()

        # safe int conversion via dict
        chart_data['home_pie'] = [
            {'value': int(counts_recent.get(0, 0)), 'name': 'Normal'},
            {'value': int(counts_recent.get(1, 0)), 'name': 'Offensive'},
            {'value': int(counts_recent.get(2, 0)), 'name': 'Hate Speech'},
        ]

        chart_data['analytics_bar'] = [
            int(counts_all.get(0, 0)),
            int(counts_all.get(1, 0)),
            int(counts_all.get(2, 0)),
        ]

    except Exception as e:
        print(f"Chart Data Error: {e}")

def save_to_db(text, lang, label, score, latency):
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute(
            '''
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                text TEXT,
                lang TEXT,
                predicted_label INTEGER,
                score REAL,
                model_name TEXT,
                latency_ms INTEGER,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            '''
        )
        cursor.execute(
            '''
            INSERT INTO predictions (text, lang, predicted_label, score, model_name, latency_ms)
            VALUES (?, ?, ?, ?, ?, ?)
            ''',
            (text, lang, int(label), float(score), 'BiLSTM', int(latency * 1000)),
        )
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"DB Save Error: {e}")

def get_history_data():
    try:
        conn = get_db_connection()
        df = pd.read_sql_query(
            "SELECT id, text, predicted_label, score, latency_ms, created_at "
            "FROM predictions ORDER BY id DESC LIMIT 50",
            conn,
        )
        conn.close()
        if df.empty:
            return []

        df['predicted_label'] = df['predicted_label'].map(
            {0: 'Normal', 1: 'Offensive', 2: 'Hate'}
        )
        df['score'] = df['score'].apply(lambda x: f"{x:.1%}")
        df['latency_ms'] = df['latency_ms'].apply(lambda x: f"{x} ms")
        df['created_at'] = df['created_at'].astype(str)
        return df.to_dict('records')
    except Exception as e:
        print(f"DB Error: {e}")
        return []

# --- 3. ADMIN FUNCTIONS ---
async def handle_retrain(log_area):
    log_area.push("🚀 Starting Retraining Process...")
    log_area.push("⏳ Please wait, this may take 1-2 minutes...")
    await asyncio.get_event_loop().run_in_executor(None, run_training)
    log_area.push("✅ Training Complete! Metrics saved to reports/.")
    ui.notify('Model Retrained Successfully!', type='positive')

async def handle_upload(e: events.UploadEventArguments):
    try:
        uploaded = e.file
        filename = uploaded.name

        # Save uploaded file to project root (or temp)
        temp_path = os.path.join(os.getcwd(), filename)
        await uploaded.save(temp_path)

        # Read CSV using pandas
        new_df = pd.read_csv(temp_path)

        # Validation
        required_cols = {'text', 'label'}
        if not required_cols.issubset(new_df.columns):
            ui.notify(
                f"❌ CSV must contain columns {required_cols}",
                type='negative'
            )
            os.remove(temp_path)
            return

        # Auto-fill lang column if missing
        if 'lang' not in new_df.columns:
            new_df['lang'] = 'en'

        # Append to clean_data.csv
        clean_path = 'data/clean_data.csv'
        new_df.to_csv(
            clean_path,
            mode='a',
            header=not os.path.exists(clean_path),
            index=False
        )

        # Cleanup temp file
        os.remove(temp_path)

        ui.notify(
            f"✅ Successfully added {len(new_df)} rows from {filename}",
            type='positive'
        )

    except Exception as err:
        ui.notify(f"❌ Upload Failed: {err}", type='negative')
        print("ADMIN UPLOAD ERROR:", err)

# --- 4. UI PAGES ---

@ui.page('/admin')
def admin_page():
    with ui.column().classes('w-full items-center p-10'):
        ui.label('🔒 KRIXION Admin Panel').classes('text-3xl font-bold mb-8 text-gray-700 dark:text-gray-200')

        pwd = ui.input('Enter Admin Password', password=True).classes('w-64')
        content_div = ui.column().classes('w-full items-center hidden')

        def check_pwd():
            if pwd.value == 'admin123':
                content_div.classes(remove='hidden')
                pwd.classes(add='hidden')
                ui.notify('Access Granted', type='positive')
            else:
                ui.notify('Access Denied', type='negative')

        ui.button('Login', on_click=check_pwd)

        with content_div:
            with ui.row().classes('w-full gap-8 justify-center'):
                # Upload card
                with ui.card().classes('p-6 w-96'):
                    ui.label('📂 Upload Dataset').classes('text-xl font-bold mb-4')
                    ui.upload(
                        label='Upload CSV (text, label)', 
                        on_upload=handle_upload,  # <--- Connects to the function above
                        auto_upload=True
                    ).props('accept=.csv')

                # Retrain card
                with ui.card().classes('p-6 w-96'):
                    ui.label('⚙️ System Maintenance').classes('text-xl font-bold mb-4')
                    log_area = ui.log().classes('w-full h-32 bg-gray-100 dark:bg-gray-800 p-2 rounded text-xs text-gray-900 dark:text-gray-100')
                    ui.button(
                        'Trigger Retraining',
                        on_click=lambda: handle_retrain(log_area),
                    ).classes('w-full bg-red-600 text-white mt-2')

            # Metrics card
            with ui.card().classes('p-6 w-full max-w-4xl mt-8'):
                ui.label('📊 Current Model Metrics').classes('text-xl font-bold mb-4')
                try:
                    with open("reports/training_report_all.json", "r") as f:
                        metrics = json.load(f)
                    ui.json_editor({'content': {'json': metrics}}).classes('w-full')
                except Exception:
                    ui.label("No metrics found.").classes('text-red-500')

# --- MAIN APP ROUTE ---
@ui.page('/')
def main_page():

    dark_mode_enabled = app.storage.user.get('dark_mode', False)
    dark_mode = ui.dark_mode(value=dark_mode_enabled)

    def toggle_theme(e):
        app.storage.user['dark_mode'] = e.value
        if e.value:
            dark_mode.enable()
        else:
            dark_mode.disable()


    ui.add_head_html("""
    <style>
    :root {
        --bg-primary: #f8fafc;
        --bg-secondary: white;
        --text-primary: #1e293b;
        --text-secondary: #64748b;
        --text-muted: #94a3b8;
        --border: #e2e8f0;
        --color-normal: #22C55E;      /* Green */
        --color-offensive: #F59E0B;   /* Orange/Amber */
        --color-hate: #EF4444;        /* Red */
    }
    .dark {
        --bg-primary: #0f172a;
        --bg-secondary: #1e293b;
        --text-primary: #f8fafc;
        --text-secondary: #cbd5e1;
        --border: #334155;
        --color-normal: #22C55E;      /* Same green */
        --color-offensive: #F59E0B;   /* Same orange */
        --color-hate: #EF4444;        /* Same red */
    }
    .result-label {
        color: var(--color-normal) !important;
        transition: color 0.3s ease;
    }

    .color-normal { color: var(--color-normal) !important; }
    .color-offensive { color: var(--color-offensive) !important; }
    .color-hate { color: var(--color-hate) !important; }

    * {
        transition: all 0.3s ease;
    }
    /* TARGETED STYLING ONLY - NO UNIVERSAL OVERRIDES */
    body.dark .q-card,
    body.dark .card,
    body.dark .q-card__section {
        background: var(--bg-secondary) !important;
        color: var(--text-primary) !important;
        border-color: var(--border) !important;
    }
    body.dark textarea,
    body.dark input,
    body.dark .q-field__control,
    body.dark .q-input__container {
        background: var(--bg-secondary) !important;
        color: var(--text-primary) !important;
        border-color: var(--border) !important;
    }
    body.dark .q-table,
    body.dark .q-table__container,
    body.dark .q-table__header,
    body.dark .q-table__row,
    body.dark .q-table__cell {
        background: var(--bg-secondary) !important;
        color: var(--text-primary) !important;
        border-color: var(--border) !important;
    }
    body.dark .echarts {
        background: var(--bg-secondary) !important;
    }
    /* Light mode text colors */
    .text-gray-800, .text-gray-900 { color: var(--text-primary) !important; }
    .text-gray-500, .text-gray-400, .text-gray-300, .text-gray-600 {
        color: var(--text-secondary) !important;
    }
    /* Dark mode text colors */
    body.dark .text-gray-800,
    body.dark .text-gray-900 {
        color: var(--text-primary) !important;
    }
    body.dark .text-gray-500,
    body.dark .text-gray-400,
    body.dark .text-gray-300,
    body.dark .text-gray-600 {
        color: var(--text-secondary) !important;
    }
    </style>
    """, shared=True)



    fetch_chart_data()

    with ui.header().classes('flex justify-between items-center px-6 bg-gradient-to-r from-blue-600 to-blue-800 text-white shadow-lg'):
        ui.label('🚀 KRIXION Hate Speech Detection').classes('text-xl font-bold text-white')
        with ui.row().classes('items-center gap-2'):
            ui.icon('light_mode', color='yellow-400')
            ui.switch(
                value=dark_mode_enabled, 
                on_change=toggle_theme
            ).props('dense color=white')


            ui.icon('dark_mode', color='purple-400')

    with ui.tabs().classes('w-full') as tabs:
        home_tab = ui.tab('Home', icon='home')
        history_tab = ui.tab('History', icon='history')
        analytics_tab = ui.tab('Analytics', icon='analytics')

    with ui.tab_panels(tabs, value=home_tab).classes('w-full p-4'):

        # --- TAB 1: HOME ---
        with ui.tab_panel(home_tab):
            with ui.row().classes('w-full justify-center gap-10'):

                # Left side: input + result
                with ui.column().classes('w-full max-w-lg'):
                    ui.label('Social Media Content Analyzer').classes('text-2xl font-bold')
                    user_input = ui.textarea(placeholder='Type Hindi/English/Hinglish text...').classes('w-full min-h-[120px]')


                    with ui.card().classes('w-full items-center p-6 mt-4 shadow-lg border'):
                        result_label = ui.label('Result: Waiting...').classes('text-3xl font-bold result-label')
                        score_label = ui.label('Confidence: -').classes('text-lg text-gray-600 dark:text-gray-300 mt-1')
                        latency_label = ui.label('').classes(
                            'text-xs text-gray-300 mt-1'
                        )

                    async def on_analyze():
                        text = user_input.value
                        if not text.strip():
                            ui.notify('Please enter text', type='warning')
                            return
                        start = time.time()
                        try:
                            cleaned = clean_text(text)
                            lang = detect_language_strict(cleaned, "indo_mixed")
                            seq = model_wrapper.preprocess([cleaned])

                            # ✅ FIX: remove batch dimension
                            raw_probs = model_wrapper.model.predict(seq, verbose=0)
                            probs = raw_probs[0]            # shape (num_classes,)
                            pred_idx = int(np.argmax(probs))
                            conf = float(probs[pred_idx])
                            lat = time.time() - start

                            save_to_db(text, lang, pred_idx, conf, lat)
                            lbls = {
                                0: ("Normal", "normal"),
                                1: ("Offensive", "offensive"), 
                                2: ("Hate Speech", "hate")
                            }
                            txt, css_class = lbls.get(pred_idx, ("Unknown", "normal"))
                            result_label.set_text(f"Result: {txt}")
                            result_label.classes(
                                remove='color-normal color-offensive color-hate',
                                add=f'result-label text-3xl font-bold color-{css_class}'
                            )

                            score_label.set_text(f"Confidence: {conf:.2%}")
                            latency_label.set_text(f"Latency: {lat:.3f}s")

                            # refresh chart data
                            fetch_chart_data()
                            home_chart.options['series'][0]['data'] = chart_data['home_pie']
                            home_chart.update()

                            # ADD THESE 2 LINES:
                            table.update_rows(get_history_data())                    # Refresh history table
                            analytics_chart.options['series'][0]['data'] = chart_data['analytics_bar']  # Refresh analytics
                            analytics_chart.update()

                            ui.notify('Saved', type='positive')
                        except Exception as e:
                            ui.notify(str(e), type='negative')

                    ui.button(
                        'Analyze',
                        on_click=on_analyze,
                    ).classes('bg-blue-600 text-white px-8 mt-4')

                # Right side: pie chart
                with ui.card().classes('w-80 p-6 shadow-lg border'):
                    ui.label('Last 10 Predictions').classes(
                        'text-sm font-bold text-gray-500 uppercase'
                    )
                    home_chart = ui.echart(
                        {
                            'tooltip': {'trigger': 'item'},
                            'legend': {'bottom': '0%'},
                            'series': [
                                {
                                    'type': 'pie',
                                    'radius': ['40%', '70%'],
                                    'data': chart_data['home_pie'],
                                    'color': ['#22C55E', '#F59E0B', '#EF4444'],
                                }
                            ],
                        }
                    ).classes('h-64 w-full')

        # --- TAB 2: HISTORY ---
        with ui.tab_panel(history_tab):
            ui.label('Recent Predictions').classes('text-xl font-bold mb-4')
            
            # Define columns FIRST
            columns = [
                {'name': 'text', 'label': 'Text', 'field': 'text', 'align': 'left'},
                {'name': 'predicted_label', 'label': 'Result', 'field': 'predicted_label'},
                {'name': 'score', 'label': 'Confidence', 'field': 'score'},
                {'name': 'latency_ms', 'label': 'Latency', 'field': 'latency_ms'},
                {'name': 'created_at', 'label': 'Timestamp', 'field': 'created_at'},
            ]
            
            # Create table with proper parameters
            table = ui.table(
                columns=columns, 
                rows=get_history_data(), 
                row_key='id'
            ).classes('w-full shadow-xl bg-transparent dark:bg-gray-900')

            def refresh_history():
                table.update_rows(get_history_data())    

            ui.button('Refresh', on_click=refresh_history).classes('mt-4 bg-blue-600 text-white px-4 py-2 rounded')


        # --- TAB 3: ANALYTICS ---
        with ui.tab_panel(analytics_tab):
            ui.label('Real-Time Analytics').classes('text-xl font-bold')

            with ui.row().classes('w-full gap-8'):
                # Bar chart
                with ui.card().classes('w-1/2 p-4 shadow-lg border'):
                    ui.label('Class Distribution (All Time)').classes(
                        'text-lg font-bold mb-2'
                    )
                    analytics_chart = ui.echart(
                        {
                            'xAxis': {
                                'type': 'category',
                                'data': ['Normal', 'Offensive', 'Hate'],
                                'axisLabel': {
                                    'color': '#64748b',  
                                    'fontSize': 12,
                                },
                                'axisLine': {
                                    'lineStyle': {
                                        'color': '#475569',  
                                    }
                                }
                            },

                            'yAxis': {
                                'type': 'value',
                                'axisLabel': {
                                    'color': '#64748b',
                                },
                                'splitLine': {
                                    'lineStyle': {
                                        'color': '#334155',
                                    }
                                }
                            },

                            'series': [
                                {
                                    'data': chart_data['analytics_bar'],
                                    'type': 'bar',
                                    'itemStyle': {'color': '#3B82F6'},
                                }
                            ],
                        }
                    ).classes('h-80 w-full')

                # Confusion matrix image
                with ui.card().classes('w-1/3 p-4 items-center border'):
                    ui.label('Model Accuracy (Test Set)').classes(
                        'text-lg font-bold mb-2'
                    )
                    if os.path.exists("reports/confusion_matrix_bilstm.png"):
                        ui.image("reports/confusion_matrix_bilstm.png").classes(
                            'w-64 rounded'
                        )
                    else:
                        ui.label('No Matrix Available')

ui.run(title='KRIXION AI Detector', reload=False, port=8080, storage_secret='krixion_hatespeech_secret')
