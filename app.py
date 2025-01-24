from dash import Dash, html, dcc, Input, Output, State, callback_context, MATCH, ALL
import dash_daq as daq
from functions.FODE_PECE import FODE_PECE, evaluate_functions
import orjson
import base64
import numpy as np
import webbrowser
import os
from threading import Timer
from math import floor

# Tworzenie aplikacji Dash
app = Dash(__name__)
app.css.config.serve_locally = True
app.scripts.config.serve_locally = True

t = None
y = None
fy = None
val_picker = {}

error_messages = {
    't_0': 'Wartość pola t₀ jest niepoprawna. Powinna być w formacie liczbowym.',
    'T': 'Wartość pola T jest niepoprawna. Powinna być w formacie liczbowym.',
    'h': 'Wartość pola h jest niepoprawna. Powinna być w formacie liczbowym.',
    'μ': 'Wartość pola μ jest niepoprawna. Powinna być w formacie liczbowym, lub pozostawione puste.',
    'μ_tol': 'Wartość pola μ_tol jest niepoprawna. Powinna być w formacie liczbowym, lub pozostawione puste.',
    'alpha': 'Wartość pola alpha dla funkcji {index} jest niepoprawna. Powinna być w formacie liczbowym.',
    'f': 'Wartość pola f dla funkcji {index} jest niepoprawna. Powinna być tekstem opisującym funkcję.',
    'y0': 'Wartość pola y₀ dla funkcji {index} jest niepoprawna. Powinna być w formacie liczbowym, rozdzielonym przecinkami.',
    'empty': "Wgrywany plik jest pusty lub ma niepoprawny format.",
    'parameters': "Brak pola parameters w pliku, ładowanie zakończone błędem.",
    'functions': "Brak pola functions w pliku, ładowanie zakończone błędem.",
    't0>T': "Krok startowy t0 jest większy lub równy wartości w kroku końcowym T",
    'alpha<0': "Wartość pola alpha dla funkcji {index} musi być większa od zera",
    'badfunc': "Funkcje są nieobliczalne, sprawdź pola i popraw",
    'y0len': "Ilości y0 nie zgadzają się, powinno ich być tyle ile ⌊max(alpha) + 1⌋"
}

app.layout = html.Div([

    # Store for triggering graph updates
    dcc.Store(id='shared-store', data={'trigger': False}),

    # Navbar
    html.Div("Diffrac", id="navbar", className="navbar"),

    # Error message
    html.Div(id="error-message", className="error hidden", children="Błąd"),

    # Main containers
    html.Div([
        # Parameter fields
        html.Div([
            html.Div([
                html.Label(['t₀ = ', dcc.Input(id='t_0', type='number', className='input-field')]),
                html.Label(['tₙ = ', dcc.Input(id='T', type='number', className='input-field')]),
                html.Label(['h = ', dcc.Input(id='h', type='number', className='input-field')]),
                html.Label(['μ = ', dcc.Input(id='μ', type='number', className='input-field', value=1)]),
                html.Label(['μₜₒₗ = ', dcc.Input(id='μ_tol', type='number', className='input-field', value=1e-6)]),
            ], className="parameter-row"),
        ], id='parameters', className="parameters-container"),

        # Calculations container
        html.Div([
            html.Div([
                html.Label([
                    html.Span('D', className="operator-label"),
                    dcc.Input(id={'type': 'alpha', 'index': 0}, type='number', className='calculation-input'),
                    html.Span(f'y0(t) =', className="operator-label")
                ], className="calculation-row"),
                dcc.Input(id={'type': 'function', 'index': 0}, type='text', className='function-input'),
                html.Label([
                    html.Span('y0(t₀) = ', className="operator-label"),
                    dcc.Input(id={'type': 'yt0', 'index': 0}, type='text', className='initial-condition-input')
                ], className="calculation-row"),
            ], id={'type': 'calculation', 'index': 0}, className="calculation-container")
        ], id='calculations', className="calculations-container"),

        # Additional calculation controls
        html.Div([
            html.Div([
                html.Button('Dodaj pole obliczeń', id='add_calculation', className='button'),
                html.Button('Usuń pole obliczeń', id='remove_calculation', className='button'),
                html.Button('Oblicz', id='button_calculate', className='button'),
            ], id='calculation_mod', className="calculation-mod-container"),

            html.Div([
                dcc.Upload(id='upload-json', children=html.Button('Importuj', id='import_button', className='button')),
                html.Button('Eksportuj', id='export_button', className='button'),
                dcc.Download(id="download-json")
            ], id='importexport', className="import-export-container"),
        ], id='calculation_other', className="additional-controls-container"),
    ], id='calculation_container', className='main-container'),

    # Plot container
    html.Div([
        dcc.Graph(id='plot-show', className="plot-graph"),
        html.Div([
            html.Div([
                html.Label("Tryb wykresu:", className="plot-label"),
                dcc.Dropdown(
                    id='plot-mode',
                    options=[{'label': '2D', 'value': '2d'}, {'label': '3D', 'value': '3d'}],
                    value='2d',
                    className='dropdown'
                )
            ], className="plot-option-row"),

            html.Div([
                html.Label("Główny tytuł wykresu:", className="plot-label"),
                dcc.Input(id='plot-title', type='text', value='Mój Wykres', className='input-field')
            ], className="plot-option-row"),

            html.Div([
                html.Label("Tytuł osi X:", className="plot-label"),
                dcc.Input(id='x-axis-title', type='text', value='Czas (t)', className='input-field'),
                html.Label("Zakres osi X:", className="plot-label"),
                html.Div([
                    dcc.Input(id='x-min', type='number', className='axis-range-input'),
                    dcc.Input(id='x-max', type='number', className='axis-range-input')
                ], className="axis-range-row")
            ], className="plot-option-row"),

            html.Div([
                html.Label("Tytuł osi Y:", className="plot-label"),
                dcc.Input(id='y-axis-title', type='text', value='Wartości', className='input-field'),
                html.Label("Zakres osi Y:", className="plot-label"),
                html.Div([
                    dcc.Input(id='y-min', type='number', className='axis-range-input'),
                    dcc.Input(id='y-max', type='number', className='axis-range-input')
                ], className="axis-range-row")
            ], className="plot-option-row"),

            html.Div([
                html.Label("Tytuł osi Z:", className="plot-label"),
                dcc.Input(id='z-axis-title', type='text', value='Wartości 2', className='input-field'),
                html.Label("Zakres osi Z:", className="plot-label"),
                html.Div([
                    dcc.Input(id='z-min', type='number', className='axis-range-input'),
                    dcc.Input(id='z-max', type='number', className='axis-range-input')
                ], className="axis-range-row")
            ], id='z-axis-options', className="plot-option-row hidden"),

            html.Div([
                html.Label("Pokaż legendę:", className="plot-label"),
                dcc.Checklist(
                    id='legend-toggle',
                    options=[{'label': 'Włącz', 'value': 'show'}],
                    value=['show'],
                    className='checklist'
                )
            ], className="plot-option-row")
        ], className="plot-options-container"),

        html.Div([
            html.Label("Zarządzaj punktami wykresu:", className="plot-label"),
            html.Button("Dodaj punkt", id="add-point-btn", className="button"),
            html.Button("Usuń punkt", id="remove-point-btn", className="button"),
            html.Div(id="plot-points-container", className="plot-points-container", children=[
                html.Div([
                    html.Div([
                        html.Label("Wartość krzywej na osi X:", className="plot-point-label"),
                        dcc.Dropdown(id={"type": "x-val-picker", "index": 0}, options=[], className="dropdown")
                    ], className="plot-point-option"),

                    html.Div([
                        html.Label("Wartość krzywej na osi Y:", className="plot-point-label"),
                        dcc.Dropdown(id={"type": "y-val-picker", "index": 0}, options=[], className="dropdown")
                    ], className="plot-point-option"),

                    html.Div([
                        html.Label("Wartość krzywej na osi Z:", className="plot-point-label"),
                        dcc.Dropdown(id={"type": "z-val-picker", "index": 0}, options=[], className="dropdown")
                    ], id={"type": "z-axis-val-container", "index": 0}, className="plot-point-option hidden"),

                    html.Div([
                        html.Label("Etykieta krzywej:", className="plot-point-label"),
                        dcc.Input(id={"type": "plot-point-label", "index": 0}, type='text', className='input-field')
                    ], className="plot-point-option"),

                    html.Div([
                        html.Label("Kolor krzywej:", className="plot-point-label"),
                        html.Div([
                            html.Button("🎨", id={"type": "color-button", "index": 0}, className="color-button"),
                            daq.ColorPicker(
                                id={"type": "color-picker", "index": 0},
                                value={"hex": "#119DFF"},
                                className="color-picker hidden"
                            )
                        ], className="color-picker-container")
                    ], className="plot-point-option")
                ], className="plot-point-row", id={"type": "plot-points", "index": 0})
            ])
        ], className="plot-point-manager")
    ], id='plot_container', className='main-container'),
], className="app-container")

# Callback do dynamicznego dodawania divów do calculations
@app.callback(
    Output('calculations', 'children', allow_duplicate=True),
    Input('add_calculation', 'n_clicks'),
    Input('remove_calculation', 'n_clicks'),
    State('calculations', 'children'),
    prevent_initial_call=True
)
def update_calculations(add_clicks, remove_clicks, existing_calculations):
    ctx = callback_context
    if not ctx.triggered:
        return existing_calculations or []

    button_id = ctx.triggered[0]['prop_id'].split('.')[0]
    existing_calculations = existing_calculations or []
    num_existing = len(existing_calculations)

    # Add a new field
    if button_id == 'add_calculation':
        new_idx = num_existing
        new_div = create_calculation_div(new_idx)
        existing_calculations.append(new_div)

    # Remove the last field (if more than 1 exists)
    elif button_id == 'remove_calculation' and num_existing > 1:
        existing_calculations.pop()

    return existing_calculations

def create_calculation_div(index):
    return html.Div([
                html.Label([
                    html.Span('D', className="operator-label"),
                    dcc.Input(id={'type': 'alpha', 'index': index}, type='number', className='calculation-input'),
                    html.Span(f'y{index}(t) =', className="operator-label")
                ], className="calculation-row"),
                dcc.Input(id={'type': 'function', 'index': index}, type='text', className='function-input'),
                html.Label([
                    html.Span(f'y{index}(t₀) = ', className="operator-label"),
                    dcc.Input(id={'type': 'yt0', 'index': index}, type='text', className='initial-condition-input')
                ], className="calculation-row"),
            ], id={'type': 'calculation', 'index': index}, className="calculation-container")

@app.callback(
    Output('download-json', 'data'),
    Output("error-message", "children", allow_duplicate=True),
    Output("error-message", "className", allow_duplicate=True),
    Input('export_button', 'n_clicks'),
    State('t_0', 'value'),
    State('T', 'value'),
    State('h', 'value'),
    State('μ', 'value'),
    State('μ_tol', 'value'),
    State('calculations', 'children'),
    prevent_initial_call=True
)
def export_data(n_clicks, t0, T, h, mu, mu_tol, calculations_div):
    global t, y, fy, error_messages

    # Initialize data structure
    data = {
        'parameters': {},
        'functions': {
            'alpha': {},
            'f': {},
            'y0': {}
        },
        'values': {
            't': np.ndarray.tolist(t) if t is not None else None,
            'y': np.ndarray.tolist(y) if y is not None else None,
            'fy': np.ndarray.tolist(fy) if fy is not None else None
        }
    }

    # Validate parameters
    try:
        data['parameters']['t0'] = float(t0)
    except Exception as e:
        return None, error_messages['t_0'], "error visible"

    try:
        data['parameters']['T'] = float(T)
    except Exception as e:
        return None, error_messages['T'], "error visible"

    try:
        data['parameters']['h'] = float(h)
    except Exception:
        return None, error_messages['h'], "error visible"

    try:
        data['parameters']['mu'] = float(mu) if mu is not None else None
    except Exception:
        return None, error_messages['μ'], "error visible"

    try:
        data['parameters']['mu_tol'] = float(mu_tol) if mu_tol is not None else None
    except Exception:
        return None, error_messages['μ_tol'], "error visible"

    # Validate functions
    try:
        for idx, text in enumerate(calculations_div):
            data['functions']['alpha'][idx] = text['props']['children'][0]['props']['children'][1]['props'].get('value', None)
            if not data['functions']['alpha'][idx]:
                return None, error_messages['alpha'].format(index=idx), "error visible"

            data['functions']['f'][idx] = text['props']['children'][1]['props'].get('value', None)
            if not data['functions']['f'][idx]:
                return None, error_messages['f'].format(index=idx), "error visible"

            tmp = text['props']['children'][2]['props']['children'][1]['props'].get('value', None)
            if tmp is None:
                return None, error_messages['y0'].format(index=idx), "error visible"
            else:
                data['functions']['y0'][idx] = [float(x.strip()) for x in tmp.split(',')] if tmp else []
    except Exception as e:
        return None, f'Error: {e}', "error visible"

    print(orjson.dumps(data, option=orjson.OPT_NON_STR_KEYS))

    # If no errors, return data for download
    return dict(content=orjson.dumps(data, option=orjson.OPT_NON_STR_KEYS).decode("utf-8"), filename='calculation_data.json'), "", "error hidden"

def ensure_dict_format(component):
    # Check if the component has a `to_plotly_json` method (common for Dash components)
    if hasattr(component, "to_plotly_json"):
        component_dict = component.to_plotly_json()

        # Recursively process children if they exist
        if "children" in component_dict["props"]:
            children = component_dict["props"]["children"]

            # If children is a list, process each child; otherwise, handle single child
            if isinstance(children, list):
                component_dict["props"]["children"] = [
                    ensure_dict_format(child) if hasattr(child, "to_plotly_json") else child
                    for child in children
                ]
            elif hasattr(children, "to_plotly_json"):
                component_dict["props"]["children"] = ensure_dict_format(children)

        return component_dict

    # Return the component as-is if it doesn't have the `to_plotly_json` method
    return component

@app.callback(
    Output('upload-json', 'contents', allow_duplicate=True),
    Output('calculations', 'children', allow_duplicate=True),
    Output('plot-points-container', 'children', allow_duplicate=True),
    Output('t_0', 'value'),
    Output('T', 'value'),
    Output('h', 'value'),
    Output('μ', 'value'),
    Output('μ_tol', 'value'),
    Output("error-message", "children", allow_duplicate=True),
    Output("error-message", "className", allow_duplicate=True),
    Input('upload-json', 'contents'),
    State('calculations', 'children'),
    State('plot-points-container', 'children'),
    prevent_initial_call=True
)
def import_data(contents, existing_calculations, plot_manager):
    global t, y, fy, val_picker, error_messages
    DEFAULT_RETURN = (None, existing_calculations or [], plot_manager or [], None, None, None, None, None)

    if contents is None:
        return *DEFAULT_RETURN, error_messages['empty'], "error visible"

    # Decode the uploaded content (JSON example)
    content_type, content_string = contents.split(',')
    decoded_data = orjson.loads(base64.b64decode(content_string).decode('utf-8'))

    num_calculations_needed = len(decoded_data['functions'].get('alpha', [1]))
    num_existing_calculations = len(existing_calculations)

    if num_calculations_needed > num_existing_calculations:
        for i in range(num_existing_calculations, num_calculations_needed):
            new_div = create_calculation_div(i)
            existing_calculations.append(new_div)
    elif num_calculations_needed < num_existing_calculations:
        for _ in range(num_existing_calculations - num_calculations_needed):
            existing_calculations.pop()

    existing_calculations = [ensure_dict_format(c) for c in existing_calculations]

    try:
        try:
            t_0 = decoded_data['parameters']['t0']
        except Exception:
            return *DEFAULT_RETURN, error_messages['t0'], "error visible"

        try:
            T = decoded_data['parameters']['T']
        except Exception:
            return *DEFAULT_RETURN, error_messages['T'], "error visible"

        try:
            h = decoded_data['parameters']['h']
        except Exception:
            return *DEFAULT_RETURN, error_messages['h'], "error visible"

        try:
            mu = decoded_data['parameters']['mu']
        except Exception:
            return *DEFAULT_RETURN, error_messages['μ'], "error visible"

        try:
            mu_tol = decoded_data['parameters']['mu_tol']
        except Exception:
            return *DEFAULT_RETURN, error_messages['μ_tol'], "error visible"

    except Exception:
        return *DEFAULT_RETURN, error_messages['parameters'], "error visible"

    try:
        tmp = decoded_data['functions']
    except Exception:
        return *DEFAULT_RETURN, error_messages['functions'], "error visible"

    for i, calculation_div in enumerate(existing_calculations):
        try:
            alpha_value = decoded_data['functions']['alpha'][str(i)]
        except Exception:
            return *DEFAULT_RETURN, error_messages['alpha'].format(index=i), "error visible"

        try:
            function_value = decoded_data['functions']['f'][str(i)]
        except Exception:
            return *DEFAULT_RETURN, error_messages['f'].format(index=i), "error visible"

        try:
            y0_value = ', '.join(map(str, decoded_data['functions']['y0'][str(i)]))
        except Exception:
            return *DEFAULT_RETURN, error_messages['y0'].format(index=i), "error visible"

        # Update the specific input fields in the calculation div
        calculation_div['props']['children'][0]['props']['children'][1]['props']['value'] = alpha_value
        calculation_div['props']['children'][1]['props']['value'] = function_value
        calculation_div['props']['children'][2]['props']['children'][1]['props']['value'] = y0_value

    try:
        t = np.array(decoded_data['values']['t'])
        y = np.array(decoded_data['values']['y'])
        fy = np.array(decoded_data['values']['fy'])
        val_picker = {'t': t}
        for i, val in enumerate(y):
            val_picker[f'y{i}'] = val
        for i, val in enumerate(fy):
            val_picker[f'fy{i}'] = val

        for div in plot_manager:
            div['props']['children'][0]['props']['children'][1]['props']['options'] = [{'label': key, 'value': key} for key in val_picker.keys()]
            div['props']['children'][1]['props']['children'][1]['props']['options'] = [{'label': key, 'value': key} for key in val_picker.keys()]
            div['props']['children'][2]['props']['children'][1]['props']['options'] = [{'label': key, 'value': key} for key in val_picker.keys()]

    except Exception:
        t = None
        y = None
        fy = None
        val_picker = None

    return None, existing_calculations, plot_manager, t_0, T, h, mu, mu_tol, "", "error hidden"

@app.callback(
Output('plot-points-container', 'children', allow_duplicate=True),
    Output("error-message", "children", allow_duplicate=True),
    Output("error-message", "className", allow_duplicate=True),
    Input('button_calculate', 'n_clicks'),
    State('t_0', 'value'),
    State('T', 'value'),
    State('h', 'value'),
    State('μ', 'value'),
    State('μ_tol', 'value'),
    State('calculations', 'children'),
    State('plot-points-container', 'children'),
    prevent_initial_call=True
)
def calculate(n_clicks, t0, T, h, mu, mu_tol, calculations_div, plot_manager):
    global t, y, fy, val_picker, error_messages

    try:
        try:
            cal_t0 = float(t0)
        except Exception:
            return plot_manager or [], error_messages['t_0'], "error visible"
        try:
            cal_T = float(T)
        except Exception:
            return plot_manager or [], error_messages['T'], "error visible"
        try:
            cal_h = float(h)
        except Exception:
            return plot_manager or [], error_messages['h'], "error visible"
        try:
            cal_mu = int(mu) if mu is not None else 1
        except Exception:
            return plot_manager or [], error_messages['μ'], "error visible"
        try:
            cal_mutol = float(mu_tol) if mu_tol is not None else 1e-6
        except Exception:
            return plot_manager or [], error_messages['μ_tol'], "error visible"

        cal_funcs, cal_alpha, cal_y0 = [], [], []

        for idx, calculation_div in enumerate(calculations_div):
            try:
                tmp = calculation_div['props']['children'][0]['props']['children'][1]['props'].get('value', None)
                if tmp == None:
                    raise ValueError
                cal_alpha.append(tmp)
            except Exception:
                return plot_manager or [], error_messages['alpha'].format(index=idx), "error visible"

            try:
                tmp = calculation_div['props']['children'][1]['props'].get('value', None)
                if tmp == None:
                    raise ValueError
                cal_funcs.append(tmp)
            except Exception:
                return plot_manager or [], error_messages['f'].format(index=idx), "error visible"

            try:
                tmp = [float(x.strip()) for x in calculation_div['props']['children'][2]['props']['children'][1]['props'].get('value', None).split(',')] if tmp else []
                if not tmp:
                    raise ValueError
                cal_y0.append(tmp)
            except Exception:
                return plot_manager or [], error_messages['y0'].format(index=idx), "error visible"

    except Exception as e:
        return plot_manager or [], f'Error: {e}', "error visible"

    cal_y0 = np.asarray(cal_y0)

    # Czas jest niepoprawny
    if t0 >= T:
        return plot_manager or [], error_messages['t0>T'], "error visible"

    # Alpha jest niepoprawna
    for idx, alpha in enumerate(cal_alpha):
        if alpha <= 0:
            return plot_manager or [], error_messages['alpha<0'].format(index=idx), "error visible"

    # Długość któregoś y0 nie jest równa max alpha
    maxalpha = floor(max(cal_alpha) + 1)
    for ty0 in cal_y0:
        if len(ty0) != maxalpha:
            return plot_manager or [], error_messages['y0len'], "error visible"

    # Nie da się obliczyć funkcji
    try:
        funcs_result = evaluate_functions(cal_funcs, cal_y0[:, 0], cal_t0)
    except Exception:
        return plot_manager or [], error_messages['badfunc'], "error visible"

    t, y, fy = FODE_PECE(cal_alpha, cal_funcs, cal_t0, cal_T, cal_y0, cal_h, cal_mu, cal_mutol)
    val_picker = {'t': t}
    for i, val in enumerate(y):
        val_picker[f'y{i}'] = val
    for i, val in enumerate(fy):
        val_picker[f'fy{i}'] = val

    for div in plot_manager:
        div['props']['children'][0]['props']['children'][1]['props']['options'] = [{'label': key, 'value': key} for key
                                                                                   in val_picker.keys()]
        div['props']['children'][1]['props']['children'][1]['props']['options'] = [{'label': key, 'value': key} for key
                                                                                   in val_picker.keys()]
        div['props']['children'][2]['props']['children'][1]['props']['options'] = [{'label': key, 'value': key} for key
                                                                                   in val_picker.keys()]

    return plot_manager, '', "error hidden"

@app.callback(
    Output({"type": "color-picker", "index": MATCH}, "style"),
    Input({"type": "color-button", "index": MATCH}, "n_clicks"),
    State({"type": "color-picker", "index": MATCH}, "style")
)
def toggle_color_picker(n_clicks, style):
    if n_clicks and style['display'] == 'none':
        return {'display': 'block', 'position': 'absolute', 'zIndex': 1000}
    return {'display': 'none', 'position': 'absolute', 'zIndex': 1000}


@app.callback(
    [
        Output('z-axis-options', 'className', allow_duplicate=True),
        Output('plot-points-container', 'children', allow_duplicate=True)
    ],
    [Input('plot-mode', 'value')],
    [State('plot-points-container', 'children')],
    prevent_initial_call=True
)
def update_plot_mode(plot_mode, points):
    # Config for the 2D or 3D plot
    if plot_mode == '2d':
        z_axis = "plot-option-row hidden"  # Hide Z-axis options
        for point in points:
            # Hide Z-axis for each point
            point['props']['children'][2]['props']['className'] = "plot-point-option hidden"
    else:
        z_axis = "plot-option-row"  # Show Z-axis options
        for point in points:
            # Show Z-axis for each point
            point['props']['children'][2]['props']['className'] = "plot-point-option"

    return z_axis, points

@app.callback(
    Output('plot-show', 'figure'),  # To update the plot's figure
    [
        Input('plot-title', 'value'),
        Input('x-axis-title', 'value'),
        Input('x-min', 'value'),
        Input('x-max', 'value'),
        Input('y-axis-title', 'value'),
        Input('y-min', 'value'),
        Input('y-max', 'value'),
        Input('z-axis-title', 'value'),
        Input('z-min', 'value'),
        Input('z-max', 'value'),
        Input('legend-toggle', 'value'),
        Input('plot-mode', 'value'),  # To determine if 3D
        Input("plot-points-container", "children"),
        Input("shared-store", "data")
    ],
    State("plot-show", "figure"),

)
def update_plot(
    plot_title, x_title, x_min, x_max, y_title, y_min, y_max, z_title, z_min, z_max, legend, plot_mode, plot_points,  store, plot_state
):
    def is_numeric(value):
        try:
            return isinstance(value, (int, float)) and not isinstance(value, bool)
        except:
            return False

    x_range = [x_min if is_numeric(x_min) else None, x_max if is_numeric(x_max) else None]
    y_range = [y_min if is_numeric(y_min) else None, y_max if is_numeric(y_max) else None]
    z_range = [z_min if is_numeric(z_min) else None, z_max if is_numeric(z_max) else None]
    is3d = (plot_mode == '3d')

    # Placeholder for data
    try:
        data = plot_state['data']
        if data[0]['x'] == [0]:
            raise Exception
    except Exception:
        if is3d:
            data = [{
                'type': 'scatter3d',
                'x': [0], 'y': [0], 'z': [0],  # Placeholder data for 3D
                'mode': 'markers',
                'marker': {'size': 12}
            }]
        else:
            data = [{
                'type': 'scatter',
                'x': [0], 'y': [0],  # Placeholder data for 2D
                'mode': 'markers',
                'marker': {'size': 12}
            }]

    graph_data = []
    for idx, div in enumerate(plot_points):
        try:
            x_val = div['props']['children'][0]['props']['children'][1]['props'].get('value', None)
            y_val = div['props']['children'][1]['props']['children'][1]['props'].get('value', None)
            if is3d:
                z_val = div['props']['children'][2]['props']['children'][1]['props'].get('value', None)
            label_val = div['props']['children'][3]['props']['children'][1]['props'].get('value', str(idx))
            color_val = div['props']['children'][4]['props']['children'][1]['props']['children'][1]['props']['value']['hex']

            # Retrieve points from val_picker
            x_points = val_picker[x_val]
            y_points = val_picker[y_val]
            if is3d:
                z_points = val_picker[z_val]

            # Determine the plot type based on plot mode
            plot_type = 'scatter3d' if plot_mode == '3d' else 'scatter'

            # Generate trace for the graph
            trace = {
                'x': x_points,
                'y': y_points,
                'mode': 'markers',
                'type': plot_type,
                'marker': {'color': color_val},
                'name': label_val  # Use label or fallback to default name
            }

            if is3d:
                trace['z'] = z_points

            graph_data.append(trace)
        except Exception:
            pass
    data = graph_data if graph_data is not None and graph_data else data

    # Base plot configuration
    fig = {
        'data': data,  # Empty for now, you will populate this based on your points
        'layout': {
            'title': plot_title,
            'xaxis': {'title': x_title, 'range': x_range},
            'yaxis': {'title': y_title, 'range': y_range},
            'showlegend': 'show' in legend,  # Enable/disable legend based on input
        }
    }
    if is3d:
        fig['layout']['scene'] = {'xaxis': {'title': x_title, 'range': x_range},
                                  'yaxis': {'title': y_title, 'range': y_range},
                                  'zaxis': {'title': z_title, 'range': z_range}}

    return fig

@app.callback(
    Output('plot-points-container', 'children', allow_duplicate=True),
    Input('add-point-btn', 'n_clicks'),
    Input('remove-point-btn', 'n_clicks'),
    State('plot-points-container', 'children'),
    State('plot-mode', 'value'),  # To determine if 3D
    prevent_initial_call=True
)
def update_plot_points(add_clicks, remove_clicks, existing_points, plot_mode):
    ctx = callback_context
    if not ctx.triggered:
        return existing_points or []

    is3d = (plot_mode == '3d')

    button_id = ctx.triggered[0]['prop_id'].split('.')[0]
    existing_points = existing_points or []
    num_existing = len(existing_points)

    # Add a new plot point set div
    if button_id == 'add-point-btn':
        new_idx = num_existing  # The index for the new plot point
        new_div = create_plot_point_div(new_idx, is3d)
        existing_points.append(new_div)

    # Remove the last plot point set div (if more than 1 exists)
    elif button_id == 'remove-point-btn' and num_existing > 1:
        existing_points.pop()

    return existing_points

def create_plot_point_div(index, is3d):
    global val_picker

    return html.Div([
                    html.Div([
                        html.Label("Wartość krzywej na osi X:", className="plot-point-label"),
                        dcc.Dropdown(id={"type": "x-val-picker", "index": index}, options=[{'label': key, 'value': key} for key
                                                                                   in val_picker.keys()], className="dropdown")
                    ], className="plot-point-option"),

                    html.Div([
                        html.Label("Wartość krzywej na osi Y:", className="plot-point-label"),
                        dcc.Dropdown(id={"type": "y-val-picker", "index": index}, options=[{'label': key, 'value': key} for key
                                                                                   in val_picker.keys()], className="dropdown")
                    ], className="plot-point-option"),

                    html.Div([
                        html.Label("Wartość krzywej na osi Z:", className="plot-point-label"),
                        dcc.Dropdown(id={"type": "z-val-picker", "index": index}, options=[{'label': key, 'value': key} for key
                                                                                   in val_picker.keys()], className="dropdown")
                    ], id={"type": "z-axis-val-container", "index": index}, className="plot-point-option hidden" if not is3d else "plot-point-option"),

                    html.Div([
                        html.Label("Etykieta krzywej:", className="plot-point-label"),
                        dcc.Input(id={"type": "plot-point-label", "index": index}, type='text', className='input-field')
                    ], className="plot-point-option"),

                    html.Div([
                        html.Label("Kolor krzywej:", className="plot-point-label"),
                        html.Div([
                            html.Button("🎨", id={"type": "color-button", "index": index}, className="color-button"),
                            daq.ColorPicker(
                                id={"type": "color-picker", "index": index},
                                value={"hex": "#119DFF"},
                                className="color-picker hidden"
                            )
                        ], className="color-picker-container")
                    ], className="plot-point-option")
                ], className="plot-point-row", id={"type": "plot-points", "index": index})

@app.callback(
    Output("shared-store", "data"),
    [
        Input({"type": "x-val-picker", "index": ALL}, "value"),
        Input({"type": "y-val-picker", "index": ALL}, "value"),
        Input({"type": "z-val-picker", "index": ALL}, "value"),
        Input({"type": "plot-point-label", "index": ALL}, "value"),
        Input({"type": "color-picker", "index": ALL}, "value")
    ],
    State("shared-store", "data")
)
def toggle_store_trigger(x_vals, y_vals, z_vals, labels, colors, current_store):
    # Toggle the store's trigger
    current_trigger = current_store.get("trigger", False)
    current_store["trigger"] = not current_trigger
    return current_store

def open_browser():
    if not os.environ.get("WERKZEUG_RUN_MAIN"):
        webbrowser.open_new('http://127.0.0.1:8050/')

if __name__ == "__main__":
    Timer(1, open_browser).start()
    app.run_server(debug=False, port=8050)

