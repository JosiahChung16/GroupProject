from shiny import App, render, ui, reactive
import pandas as pd
import os
import json

from htmltools import HTML
import plotly.graph_objects as go

# base paths for datasets
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEATHS_PATH = os.path.join(BASE_DIR, "Datasets", "Deaths")
SOCIAL_PATH = os.path.join(BASE_DIR, "Datasets", "Social")
GEOJSON_PATH = os.path.join(BASE_DIR, "Datasets", "geojson-counties-fips.json")

# Map dataset for Georgia
with open(GEOJSON_PATH) as feature:
    ga_map = json.load(feature)
georgia_features = [feature for feature in ga_map["features"] if feature["id"].startswith("13")]
ga_map["features"] = georgia_features

# death dataset

deaths_datasets = {
    "All":             os.path.join(DEATHS_PATH, "HDPulse_data_All.csv"),
    "Accidents":       os.path.join(DEATHS_PATH, "HDPulse_data_Accidents.csv"),
    "Alzheimers":      os.path.join(DEATHS_PATH, "HDPulse_data_Alzheimers.csv"),
    "Cancer":          os.path.join(DEATHS_PATH, "HDPulse_data_Cancer.csv"),
    "Cerebrovascular": os.path.join(DEATHS_PATH, "HDPulse_data_Cerebrovascular.csv"),
    "Diabetes":        os.path.join(DEATHS_PATH, "HDPulse_data_Diabetes.csv"),
    "Heart":           os.path.join(DEATHS_PATH, "HDPulse_data_Heart.csv"),
    "Homicide":        os.path.join(DEATHS_PATH, "HDPulse_data_Homocide.csv"),
    "Influenza":       os.path.join(DEATHS_PATH, "HDPulse_data_Influenza.csv"),
    "Kidney":          os.path.join(DEATHS_PATH, "HDPulse_data_Kidney.csv"),
    "Liver":           os.path.join(DEATHS_PATH, "HDPulse_data_Liver.csv"),
    "Pneumonia":       os.path.join(DEATHS_PATH, "HDPulse_data_Pneumonia.csv"),
    "Respiratory":     os.path.join(DEATHS_PATH, "HDPulse_data_Respiratory.csv"),
    "Septicemia":      os.path.join(DEATHS_PATH, "HDPulse_data_Septicemia.csv"),
    "Suicide":         os.path.join(DEATHS_PATH, "HDPulse_data_Suicide.csv")
}

social_datasets = {
    "Incarceration":         os.path.join(SOCIAL_PATH, "HDPulse_data_Incarceration.csv"),
    "Income":                os.path.join(SOCIAL_PATH, "HDPulse_data_Income.csv"),
    "Non-English Language":  os.path.join(SOCIAL_PATH, "HDPulse_data_NonEnglishLanguage.csv"),
    "Poverty":               os.path.join(SOCIAL_PATH, "HDPulse_data_Poverty.csv"),
    "Smoking":               os.path.join(SOCIAL_PATH, "HDPulse_data_Smoking.csv"),
    "Under18":               os.path.join(SOCIAL_PATH, "HDPulse_data_Under18.csv"),
    "Unemployed":            os.path.join(SOCIAL_PATH, "HDPulse_data_Unemployed.csv"),
    "Age 18-39":             os.path.join(SOCIAL_PATH, "HDPulse_data_Age18_39.csv"),
    "Age 40 & Over":         os.path.join(SOCIAL_PATH, "HDPulse_data_Age40_Over.csv"),
    "Education":             os.path.join(SOCIAL_PATH, "HDPulse_data_Education.csv"),
}

# This dictionary is used by the risk lookup
types = {
    "Deaths": deaths_datasets,
    "Social": social_datasets
}

# merge all different causes into one dataset
def merge_datasets():
    merged_causes = pd.DataFrame()
    for cause, file_path in deaths_datasets.items():
        # clean up data
        df = pd.read_csv(
            file_path,
            skiprows=4,
            skipfooter=10,
            engine="python"
        )
        # Removes quotes
        df.columns = [col.strip('"') for col in df.columns]
        rate_col = f"mortality_rate_{cause.lower()}"
        df[rate_col] = pd.to_numeric(df['Age-Adjusted Death Rate(†) - deaths per 100,000'], errors='coerce')
        df = df.dropna(subset=[rate_col])

        # we need FIPS to merge the data to the map data
        df['FIPS'] = df['FIPS'].astype("Int64")

        temp_df = df[['County', 'FIPS', rate_col]].sort_values(by='County').reset_index(drop=True)
        if not merged_causes.empty:
            merged_causes = pd.merge(merged_causes, temp_df, on=['County', 'FIPS'], how='outer')
        else:
            merged_causes = temp_df
    return merged_causes

merged_causes = merge_datasets()

#load a CSV and clean first two columns
def load_and_clean(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path, skiprows=4)
    df.columns.values[0:2] = ["county_name", "fips"]
    df["fips"] = df["fips"].astype(str).str.strip()
    df["county_name"] = df["county_name"].str.lower().str.strip()
    return df


def safe_get(row, colname):
    return row[colname] if colname in row else "N/A"

### Shiny UI ###
# leftside: sidebar with dropdown menu & user instructions
# rightside: map, barchart, then patient risk lookup
app_ui = ui.page_sidebar(
    ui.sidebar(
        ui.h3("Georgia State Mortality Map"),
        ui.p("Choose a cause of death to see which Georgia counties had the highest mortality rates from 2018 to 2022."),

        # dropdown for map
        ui.input_select(
            "cause",
            "Cause of Death",
            choices=list(deaths_datasets.keys()),
            selected="All"  # default selection
        ),
        ui.p(
            "Use the dropdown menu to switch between causes. "
            "The map and chart update accordingly."
        ),
        ui.p(
            "Data Source: ",
            ui.a("HDPulse Data Portal", href="https://hdpulse.nimhd.nih.gov/data-portal/home", target="_blank")
        )
    ),

    ui.div(
        ui.output_ui("map_ui"),  # mortality map
        ui.output_ui("bar_ui"),  # top 5 bar chart
        ui.hr(),
        # Patient risk lookup at bottom
        ui.h2("County-Based Health Risk Lookup"),
        ui.input_text("first_name", "First Name", ""),
        ui.input_text("last_name", "Last Name", ""),
        ui.input_text("age", "Age"),
        ui.input_select("gender", "Gender", choices=["All Genders"]),
        ui.input_text("county", "County Name (for risk lookup and map highlight)", ""),
        ui.input_select("category", "Category", choices=list(types.keys())),
        ui.output_ui("dataset_ui"),
        ui.input_action_button("submit", "Check Risk"),
        ui.output_text_verbatim("result")  # use verbatim to preserve line breaks
    )
)

### Shiny Server ###
def server(input, output, session):
    @output
    @render.ui
    def map_ui():
        cause_of_death = input.cause() or "All"  # from dropdown menu selection, need a default
        cause_col = cause_of_death.lower()
        rate_col = f"mortality_rate_{cause_col}"

        all_df = merged_causes[['County', 'FIPS', rate_col]].copy()
        all_df = all_df.rename(columns={rate_col: 'mortality_rate'})
        valid_data = all_df.dropna(subset=['mortality_rate'])  # remove NAs
        fig = go.Figure()

        # back map - an empty map (all counties)
        # add gray borders for all counties so that can see the county with missing value
        fig.add_trace(
            go.Choropleth(
                geojson=ga_map,
                locations=all_df['FIPS'],  # merge by FIPS
                z=[0] * len(all_df),  # 0 for background
                featureidkey='id',
                colorscale=[[0, 'white'], [1, 'white']],
                showscale=False,
                marker_line_color='lightgray',
                marker_line_width=0.5,
                hoverinfo='skip'
            )
        )

        # for the map's color range
        zmin = valid_data['mortality_rate'].min()
        zmax = valid_data['mortality_rate'].max()

        # front map (counties with data)
        fig.add_trace(
            go.Choropleth(
                geojson=ga_map,
                locations=valid_data['FIPS'],
                z=valid_data['mortality_rate'],
                featureidkey='id',
                colorscale='Reds',
                zmin=zmin,
                zmax=zmax,
                colorbar_title='Deaths per 100k',
                marker_line_color='lightgray',
                marker_line_width=0.5,
                hovertext=valid_data['County'],
                hovertemplate='%{hovertext}<br>Death Rate: %{z}<extra></extra>'
            )
        )

        fig.update_geos(fitbounds='locations', visible=False)  # focus on GA
        fig.update_layout(
            title=f"{cause_of_death} Mortality by County (GA, 2018–2022)",
            height=500,
            margin={'r': 0, 't': 40, 'l': 0, 'b': 0},
            paper_bgcolor='white',
            plot_bgcolor='white'
        )

        return HTML(fig.to_html(full_html=False, include_plotlyjs='cdn'))

    @output
    @render.ui
    def bar_ui():
        cause_of_death = input.cause() or "All"  # from drop down input
        cause_col = cause_of_death.lower()
        rate_col = f"mortality_rate_{cause_col}"

        all_df = merged_causes[['County', 'FIPS', rate_col]].copy()
        all_df = all_df.rename(columns={rate_col: 'mortality_rate'})
        valid_data = all_df.dropna(subset=['mortality_rate'])  # remove NAs

        top5_counties = valid_data.sort_values(by='mortality_rate', ascending=False).head(5)  # top 5

        fig = go.Figure()
        fig.add_trace(
            go.Bar(
                x=top5_counties['mortality_rate'],
                y=top5_counties['County'],
                orientation='h',  # horizontal bar
                marker={'color': '#fcae91'},  # pink bars
                hovertemplate='%{y}<br>Death Rate: %{x}<extra></extra>'
            )
        )

        fig.update_layout(
            title=f"Top 5 GA Counties by {cause_of_death} Mortality (2018–2022)",
            height=300,
            margin={'r': 10, 't': 40, 'l': 10, 'b': 40},
            yaxis={'autorange': 'reversed'},
            plot_bgcolor='white',
            paper_bgcolor='white'
        )
        # change grid design
        fig.update_xaxes(showgrid=True, gridcolor='lightgray')
        fig.update_yaxes(showgrid=False)

        return HTML(fig.to_html(full_html=False, include_plotlyjs='cdn'))

    # Dynamically populate dataset dropdown based on category
    @output
    @render.ui
    def dataset_ui():
        cat = input.category()
        choices = sorted(types.get(cat, {}).keys())
        return ui.input_select('dataset', 'Dataset', choices=choices, selected=choices[0] if choices else None)

    # Compute patient risk lookup when user clicks 'Check Risk'
    @reactive.Calc
    def risk_info():
        if input.submit() == 0:
            return ""
        cat = input.category()
        ds = input.dataset()
        county_input = input.county().strip().lower()
        if not county_input:
            return "Please enter a county name."
        path = types.get(cat, {}).get(ds)
        if not path:
            return f"No dataset named '{ds}' found in category '{cat}'"
        try:
            df = load_and_clean(path)
        except Exception as e:
            return f"Error loading dataset '{ds}': {e}"
        match = df[df['county_name'].str.contains(county_input, na=False)]
        if match.empty:
            return f"County '{county_input}' not found in the '{ds}' dataset."
        row = match.iloc[0]
        lines = [
            f"Patient: {input.first_name()} {input.last_name()}",
            f"Age: {input.age()}, Gender: {input.gender()}",
            f"Category: {cat}",
            f"Dataset: {ds}",
            f"County: {row['county_name'].title()}"
        ]
        for col in df.columns[2:]:
            lines.append(f"{col}: {safe_get(row, col)}")
        return "\n".join(lines)

    @output
    @render.text
    def result():
        return risk_info()

app = App(app_ui, server)
