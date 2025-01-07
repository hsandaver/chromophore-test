# streamlit_app.py

import streamlit as st
import pandas as pd
import re
import warnings
import altair as alt
from io import BytesIO
from PIL import Image

# RDKit imports
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.Chem import rdMolDescriptors
from rdkit import RDLogger
from rdkit.Chem import AllChem

# For image visualization (2D)
from rdkit.Chem import Draw

# For 3D visualization
import py3Dmol

RDLogger.DisableLog('rdApp.*')  # Suppress RDKit warnings
warnings.filterwarnings("ignore")

##############################
# SMARTS Patterns Definition #
##############################

chromophore_patterns = {
    'Azo': Chem.MolFromSmarts('N=N'),
    'Anthraquinone': Chem.MolFromSmarts('C1=CC=C2C(=O)C=CC2=O'),
    'Nitro': Chem.MolFromSmarts('[NX3](=O)=O'),
    'Quinone': Chem.MolFromSmarts('O=C1C=CC=CC1=O'),
    'Indigoid': Chem.MolFromSmarts('C1C=CC(=O)NC1=O'),
    'Cyanine': Chem.MolFromSmarts('C=C-C=C'),
    'Xanthene': Chem.MolFromSmarts('O1C=CC2=C1C=CC=C2'),
    'Thiazine': Chem.MolFromSmarts('N1C=NC=S1'),
    'Coumarin': Chem.MolFromSmarts('O=C1OC=CC2=CC=CC=C12'),
    'Porphyrin': Chem.MolFromSmarts('N1C=CC=N1'),
    'Phthalocyanine': Chem.MolFromSmarts('C1=C(C2=NC=C(C)N2)C3=CC=CC=C13'),
    'Carotenoid': Chem.MolFromSmarts('C=C(C)C=CC=C'),
    'Squaraine': Chem.MolFromSmarts('C=CC=C'),
    'Metal Complex': Chem.MolFromSmarts('[!#1]'),
    'Bromine': Chem.MolFromSmarts('Br'),  
    'Selenium': Chem.MolFromSmarts('Se'),
    'Pyridine': Chem.MolFromSmarts('C1=CC=NC=C1'),
    'Phosphine': Chem.MolFromSmarts('P(C)(C)C'),
    'Carbene': Chem.MolFromSmarts('[C]')
}

auxochrome_patterns = {
    'Hydroxyl': Chem.MolFromSmarts('[OX2H]'),
    'Amine': Chem.MolFromSmarts('N'),
    'Methoxy': Chem.MolFromSmarts('COC'),
    'Thiol': Chem.MolFromSmarts('[SX2H]'),
    'Carboxyl': Chem.MolFromSmarts('C(=O)[OX2H1]')
}

#################################
# Helper Functions & Enhancements
#################################

def better_smiles_correction(smiles: str) -> tuple:
    corrected_smiles = smiles
    corrected = False

    # Remove stereochemistry markers
    if '/' in corrected_smiles or '\\' in corrected_smiles:
        corrected_smiles = re.sub(r'[\\/]', '', corrected_smiles)
        corrected = True

    # Balance ring numbers if possible (simple approach)
    ring_numbers = re.findall(r'\d', corrected_smiles)
    if len(ring_numbers) % 2 != 0:
        corrected_smiles = re.sub(r'\d', '', corrected_smiles, count=1)
        corrected = True

    # Fix common bracket mismatches
    open_brackets = corrected_smiles.count('[')
    close_brackets = corrected_smiles.count(']')
    if open_brackets > close_brackets:
        corrected_smiles = corrected_smiles.replace('[', '', (open_brackets - close_brackets))
        corrected = True
    elif close_brackets > open_brackets:
        corrected_smiles = corrected_smiles.replace(']', '', (close_brackets - open_brackets))
        corrected = True

    return corrected_smiles, corrected

def identify_chromophores(smiles: str) -> str:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return 'Invalid SMILES'
    matched_chromophores = []
    for name, pattern in chromophore_patterns.items():
        if pattern and mol.HasSubstructMatch(pattern):
            matched_chromophores.append(name)
    return ', '.join(matched_chromophores) if matched_chromophores else 'Unknown'

def identify_auxochromes(smiles: str) -> str:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return 'Invalid SMILES'
    matched_auxochromes = []
    for name, pattern in auxochrome_patterns.items():
        if pattern and mol.HasSubstructMatch(pattern):
            matched_auxochromes.append(name)
    return ', '.join(matched_auxochromes) if matched_auxochromes else 'None'

def calc_num_double_bonds(mol: Chem.Mol) -> int:
    return sum(1 for bond in mol.GetBonds() if bond.GetBondType() == Chem.rdchem.BondType.DOUBLE)

def calc_num_rings(mol: Chem.Mol) -> int:
    return mol.GetRingInfo().NumRings()

def calculate_descriptors(smiles: str) -> dict:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return {
            'MolWeight': None,
            'LogP': None,
            'TPSA': None,
            'NumRings': None,
            'NumDoubleBonds': None
        }
    try:
        mol_weight = Descriptors.MolWt(mol)  
        logP = Descriptors.MolLogP(mol)      
        tpsa = rdMolDescriptors.CalcTPSA(mol)
        num_rings = calc_num_rings(mol)
        num_double_bonds = calc_num_double_bonds(mol)
    except Exception:
        return {
            'MolWeight': None,
            'LogP': None,
            'TPSA': None,
            'NumRings': None,
            'NumDoubleBonds': None
        }
    return {
        'MolWeight': mol_weight,
        'LogP': logP,
        'TPSA': tpsa,
        'NumRings': num_rings,
        'NumDoubleBonds': num_double_bonds
    }

def chromophore_to_color(chromophore: str) -> str:
    chromophore_color_map = {
        'Azo': 'Red/Orange/Yellow',
        'Anthraquinone': 'Red/Blue/Violet',
        'Nitro': 'Yellow/Orange',
        'Quinone': 'Yellow/Orange/Brown',
        'Indigoid': 'Blue/Purple',
        'Cyanine': 'Green/Blue',
        'Xanthene': 'Yellow/Orange',
        'Thiazine': 'Blue/Green',
        'Coumarin': 'Blue/Green',
        'Porphyrin': 'Red/Purple',
        'Phthalocyanine': 'Green/Blue',
        'Carotenoid': 'Yellow/Orange',
        'Squaraine': 'Red/Purple',
        'Bromine': 'Dark Green/Purple',
        'Selenium': 'Deep Blue/Purple',
        'Pyridine': 'Varies (often Green/Blue/Yellow)',
        'Phosphine': 'Varies (Yellow/Green)',
        'Carbene': 'Varies (Red/Purple)',
        'Metal Complex': 'Varies (often Green/Blue)'
    }
    return chromophore_color_map.get(chromophore, 'Unknown')

def estimate_color(chromophores: str, auxochromes: str, descriptors: dict) -> str:
    if chromophores == 'Invalid SMILES':
        return 'Invalid SMILES'
    if chromophores == 'Unknown' and descriptors:
        try:
            if descriptors['NumDoubleBonds'] and descriptors['NumDoubleBonds'] > 5:
                base_color = 'Red/Orange'
            elif descriptors['NumDoubleBonds'] and descriptors['NumDoubleBonds'] > 3:
                base_color = 'Yellow/Orange'
            elif descriptors['NumRings'] and descriptors['NumRings'] >= 4:
                base_color = 'Blue/Violet'
            elif descriptors['NumRings'] and descriptors['NumRings'] == 3:
                base_color = 'Green/Blue'
            elif descriptors['MolWeight'] and descriptors['MolWeight'] > 600:
                base_color = 'Deep Color (likely Red or Purple)'
            elif descriptors['MolWeight'] and descriptors['MolWeight'] > 400:
                base_color = 'Moderate Color (likely Blue or Green)'
            else:
                base_color = 'Lighter Color (likely Yellow)'
        except TypeError:
            base_color = 'Unknown'
    else:
        first_chromophore = chromophores.split(', ')[0]
        base_color = chromophore_to_color(first_chromophore)

    if auxochromes != 'Invalid SMILES':
        auxo_list = [auxo.strip() for auxo in auxochromes.split(',')]
        if 'Hydroxyl' in auxo_list:
            base_color += ' (Shifted towards Red)'
        if 'Amine' in auxo_list:
            base_color += ' (Shifted towards Blue/Violet)'
        if 'Methoxy' in auxo_list:
            base_color += ' (Shifted towards Yellow)'
        if 'Thiol' in auxo_list:
            base_color += ' (Potential for Increased Color Intensity)'
        if 'Carboxyl' in auxo_list:
            base_color += ' (Potential for Increased Solubility)'

    return base_color

@st.cache_data
def process_file(df: pd.DataFrame) -> pd.DataFrame:
    if 'SMILES_Corrected' not in df.columns:
        df['SMILES_Corrected'] = False
    if 'SMILES_Valid' not in df.columns:
        df['SMILES_Valid'] = False
    if 'Corrected_SMILES' not in df.columns:
        df['Corrected_SMILES'] = df['SMILES']

    for idx, row in df.iterrows():
        smiles = row['SMILES']
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            corrected_smiles, corrected = better_smiles_correction(smiles)
            df.at[idx, 'Corrected_SMILES'] = corrected_smiles
            if corrected:
                df.at[idx, 'SMILES_Corrected'] = True
                mol = Chem.MolFromSmiles(corrected_smiles)
                df.at[idx, 'SMILES_Valid'] = bool(mol)
            else:
                df.at[idx, 'SMILES_Valid'] = False
        else:
            df.at[idx, 'SMILES_Valid'] = True

    df['Chromophore'] = df['Corrected_SMILES'].apply(identify_chromophores)
    df['Auxochrome'] = df['Corrected_SMILES'].apply(identify_auxochromes)
    df['Descriptors'] = df['Corrected_SMILES'].apply(calculate_descriptors)
    descriptor_df = df['Descriptors'].apply(pd.Series)

    for col in ['MolWeight','LogP','TPSA','NumRings','NumDoubleBonds']:
        if col not in descriptor_df.columns:
            descriptor_df[col] = None
    df = pd.concat([df, descriptor_df], axis=1)
    df.drop(columns=['Descriptors'], inplace=True)

    df['Estimated Color'] = df.apply(
        lambda x: estimate_color(
            x['Chromophore'],
            x['Auxochrome'],
            {
                'MolWeight': x['MolWeight'],
                'LogP': x['LogP'],
                'TPSA': x['TPSA'],
                'NumRings': x['NumRings'],
                'NumDoubleBonds': x['NumDoubleBonds']
            }
        ), axis=1
    )

    return df

########################
# Visualization Helpers
########################

def visualize_smiles_2d(smiles: str):
    mol = Chem.MolFromSmiles(smiles)
    if mol:
        return Draw.MolToImage(mol, size=(250, 250))
    else:
        return None

def visualize_smiles_3d(smiles: str):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return "Invalid SMILES"
    mol = Chem.AddHs(mol)
    try:
        _ = AllChem.EmbedMolecule(mol, AllChem.ETKDG())
        _ = AllChem.MMFFOptimizeMolecule(mol)
    except Exception as e:
        return f"3D embedding failed: {e}"
    mb = Chem.MolToMolBlock(mol)
    view = py3Dmol.view(width=350, height=300)
    view.addModel(mb, 'mol')
    view.setStyle({'stick': {}})
    view.zoomTo()
    return view.render()

#################
# Streamlit App #
#################

def main():
    st.set_page_config(
        page_title="Dye Color Analysis - Enhanced",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    st.title("Enhanced Chromophore & Auxochrome Analysis App")
    st.markdown(
        """
        *Nyaa~ This is the upgraded version of our adorable **Streamlit** app! 
        Now featuring:*  
        1. **Better SMILES correction**  
        2. **Single SMILES input**  
        3. **3D or 2D molecular visualization**  
        4. **Caching for quicker performance**  
        5. **Interactive distribution plots**  
        
        **How to use:**  
        1. Choose your method (upload CSV or type a single SMILES).  
        2. We’ll try to fix any quirky SMILES, find chromophores/auxochromes, 
           and guess your molecule’s color range.  
        3. We’ll show a table and let you download results if you used a CSV.  
        4. Enjoy the color-coded previews, meow~!
        """
    )

    st.sidebar.header("Select Input Method")
    method = st.sidebar.radio("Method:", ["CSV Upload", "Single SMILES Input"])
    visualize_option = st.sidebar.radio("Visualization Type:", ["2D", "3D"])

    if method == "CSV Upload":
        uploaded_file = st.file_uploader("Upload a CSV file with 'SMILES' column...", type=["csv"])
        if uploaded_file:
            try:
                df = pd.read_csv(uploaded_file)
            except Exception as e:
                st.error(f"**Oh no!** There was an error reading your file: {e}")
                return

            if 'SMILES' not in df.columns:
                st.error("**Oops!** The uploaded CSV must contain a 'SMILES' column.")
                return

            with st.spinner("Processing your CSV, nyah~..."):
                df = process_file(df)

            st.subheader("Analysis Results")
            st.markdown("Below is a table of your processed SMILES with estimated colors and descriptors, nyaa~")

            def color_background(val: str) -> str:
                text_lower = str(val).lower()
                color_string = "#ffffff"  # default
                if "red" in text_lower or "orange" in text_lower:
                    color_string = "#ffebe6"
                if "blue" in text_lower or "violet" in text_lower or "purple" in text_lower:
                    color_string = "#ebe6ff"
                if "green" in text_lower:
                    color_string = "#e6ffeb"
                if "yellow" in text_lower or "brown" in text_lower:
                    color_string = "#f9ffe6"
                return f"background-color: {color_string};"

            styled_df = df.style.applymap(color_background, subset=['Estimated Color'])
            st.dataframe(styled_df, use_container_width=True)

            st.subheader("Molecular Weight Distribution")
            chart = (
                alt.Chart(df[df['MolWeight'].notnull()])
                .mark_bar(color='#9b59b6')
                .encode(
                    alt.X('MolWeight:Q', bin=alt.Bin(maxbins=30), title='Molecular Weight'),
                    y='count()'
                )
                .properties(width=600, height=400)
            )
            st.altair_chart(chart, use_container_width=True)

            st.subheader("Molecular Previews")
            for i, row in df.iterrows():
                if row['SMILES_Valid']:
                    st.markdown(f"**Index {i}** | **Estimated Color:** {row['Estimated Color']}")
                    if visualize_option == "2D":
                        mol_image = visualize_smiles_2d(row['Corrected_SMILES'])
                        if mol_image:
                            buf = BytesIO()
                            mol_image.save(buf, format="PNG")
                            st.image(Image.open(buf))
                        else:
                            st.write("**Couldn't render 2D image.**")
                    else:
                        view_3d = visualize_smiles_3d(row['Corrected_SMILES'])
                        if isinstance(view_3d, str):
                            st.write(view_3d)
                        else:
                            st.components.v1.html(view_3d, height=300)
                else:
                    st.write(f"Index {i}: Invalid SMILES, no image to display.")

            st.subheader("Download Enhanced Data")
            st.markdown("Nyaa~ Here you can download your updated file with color estimates and corrections.")
            csv_data = df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Download CSV",
                data=csv_data,
                file_name="output_dye_colors_enhanced.csv",
                mime="text/csv"
            )
        else:
            st.info("Please upload a CSV file to begin, meow~!")

    else:
        st.subheader("Single SMILES Input")
        user_smiles = st.text_input("Enter SMILES string here, nyaa~:")
        if user_smiles:
            tmp_df = pd.DataFrame({"SMILES": [user_smiles]})
            with st.spinner("Analyzing your SMILES, nyah~"):
                result_df = process_file(tmp_df)

            row = result_df.iloc[0]
            st.write("**Corrected SMILES:**", row['Corrected_SMILES'])
            st.write("**Valid SMILES?**", row['SMILES_Valid'])
            st.write("**Chromophores:**", row['Chromophore'])
            st.write("**Auxochromes:**", row['Auxochrome'])
            st.write("**MolWeight:**", row['MolWeight'])
            st.write("**LogP:**", row['LogP'])
            st.write("**TPSA:**", row['TPSA'])
            st.write("**NumRings:**", row['NumRings'])
            st.write("**NumDoubleBonds:**", row['NumDoubleBonds'])
            st.write("**Estimated Color:**", row['Estimated Color'])

            if row['SMILES_Valid']:
                st.subheader("Structure Preview")
                if visualize_option == "2D":
                    mol_image = visualize_smiles_2d(row['Corrected_SMILES'])
                    if mol_image:
                        buf = BytesIO()
                        mol_image.save(buf, format="PNG")
                        st.image(Image.open(buf))
                    else:
                        st.write("**Couldn't render 2D image.**")
                else:
                    view_3d = visualize_smiles_3d(row['Corrected_SMILES'])
                    if isinstance(view_3d, str):
                        st.write(view_3d)
                    else:
                        st.components.v1.html(view_3d, height=300)
            else:
                st.write("**Oops, the SMILES is invalid, sorry!**")

if __name__ == "__main__":
    main()