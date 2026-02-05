# ==============================================================================
# ⚛️ OMNIKERNEL CENTURION v101.5 | UNIFIED THERMODYNAMIC EDITION
# ==============================================================================
# ARQUITECTURA : STREAMLIT SERVER-SIDE (IP SAFE)
# DESPLIEGUE   : ANDROID / WEB / DESKTOP
# PARADIGMA    : TCDS (Q · Σ = φ)
# AUTOR        : Genaro Carrasco Ozuna
# ==============================================================================

# ------------------------------------------------------------------------------
# 1. CONFIGURACIÓN DE PÁGINA (OBLIGATORIO PRIMERO)
# ------------------------------------------------------------------------------
import streamlit as st
st.set_page_config(
    page_title="OmniKernel Centurion v101.5",
    page_icon="⚛️",
    layout="wide",
    initial_sidebar_state="collapsed"
)
# =========================
# BLOQUE 1: ESTADO TERMO
# =========================

class ThermoState:
    def __init__(self, pressure=1.0, tension=0.0, temperature=300.0):
        self.pressure = pressure        # escala adimensional
        self.tension = tension          # deformación estructural
        self.temperature = temperature  # K (efectiva)

    def apply(self, coord):
        """
        Aplica presión + tensión a un vector (x,y,z)
        """
        x, y, z = coord
        scale = 1.0 - self.pressure * 0.01
        deform = 1.0 + self.tension * 0.01
        return (
            x * scale * deform,
            y * scale,
            z * scale
        )
# ------------------------------------------------------------------------------
# 2. IMPORTACIONES
# ------------------------------------------------------------------------------
import numpy as np
import io, os, time, json, zipfile
import streamlit.components.v1 as components

try:
    import py3Dmol
    from Bio.PDB import PDBParser
except ImportError as e:
    st.error(f"Dependencia faltante: {e}")
    st.stop()
# =========================
# BLOQUE 2: DEFORMACIÓN
# =========================

def apply_thermo_to_structure(structure, thermo_state):
    """
    Devuelve coordenadas deformadas sin alterar el objeto original
    """
    new_coords = []

    for atom in structure.get_atoms():
        x, y, z = atom.get_coord()
        new_coords.append(thermo_state.apply((x, y, z)))

    return new_coords
# ------------------------------------------------------------------------------
# 3. ESTILOS DEEPTECH
# ------------------------------------------------------------------------------
st.markdown("""
<style>
.stApp { background:#050505; color:#00ffcc; font-family:Courier New; }
h1,h2,h3 { border-bottom:1px solid #004444; }
button { background:#002222 !important; color:#00ffcc !important; }
</style>
""", unsafe_allow_html=True)
# =========================
# BLOQUE 3: MÉTRICAS
# =========================

def compute_energy_metrics(coords, thermo_state):
    """
    Métricas simples pero deterministas
    """
    energies = []

    for x, y, z in coords:
        r = (x**2 + y**2 + z**2) ** 0.5
        energy = thermo_state.pressure * r + thermo_state.tension * abs(z)
        energies.append(energy)

    return energies
# ==============================================================================
# 4. NÚCLEO FÍSICO TCDS
# ==============================================================================

class TCDS_ThermoEngine:
    """
    Motor termodinámico ontológico TCDS.
    No estocástico. Paramétrico.
    """

    def __init__(self, structure):
        self.structure = structure
        self.coords = self._extract_coords()

    def _extract_coords(self):
        atoms = [a for a in self.structure.get_atoms()]
        return np.array([a.get_coord() for a in atoms])

    def apply_pressure(self, sigma, q_ext):
        center = np.mean(self.coords, axis=0)
        new_coords = []

        for r in self.coords:
            vec = r - center
            dist = np.linalg.norm(vec) + 1e-6
            delta = (sigma * q_ext) / dist
            new_coords.append(r + vec/dist * delta)

        return np.array(new_coords)

    def stress_field(self, new_coords):
        return np.linalg.norm(new_coords - self.coords, axis=1)
# =========================
# BLOQUE 4: VISUALIZACIÓN
# =========================

import py3Dmol
import streamlit as st

def show_structure(pdb_string):
    view = py3Dmol.view(width=800, height=600)
    view.addModel(pdb_string, "pdb")
    view.setStyle({"cartoon": {"color": "spectrum"}})
    view.zoomTo()
    st.components.v1.html(view._make_html(), height=600)
# ==============================================================================
# 5. MOTOR ONTOLÓGICO MULTISUSTRATO
# ==============================================================================

class OntologicalEngine:

    SUBSTRATES = {
        "TIERRA":        {"sigma": 1.0},
        "LUNA":          {"sigma": 0.05},
        "MARTE":         {"sigma": 0.38},
        "SOL":           {"sigma": 55.0},
        "HIPERPRESION":  {"sigma": 120.0},
        "VACIO_INDUCIDO":{"sigma": 0.01},
        "HEXATRON":      {"sigma": 1.0}
    }

    def parse_structure(self, pdb_text):
        parser = PDBParser(QUIET=True)
        return parser.get_structure("TCDS", io.StringIO(pdb_text))

    def q_factor(self, structure):
        coords = np.array([a.get_coord() for a in structure.get_atoms()])
        center = np.mean(coords, axis=0)
        rg = np.sqrt(np.mean(np.sum((coords-center)**2, axis=1)))
        return (100/(rg+1e-6))*6.5
# =========================
# BLOQUE 5: EXPORT PDB
# =========================

def export_pdb(structure, coords, filepath):
    out = ""
    atom_serial = 1

    for atom, (x, y, z) in zip(structure.get_atoms(), coords):
        res = atom.get_parent()
        chain = res.get_parent()

        atom_name = atom.get_name()
        resname = res.get_resname()
        chain_id = chain.id if chain.id.strip() else "A"
        resseq = res.id[1]
        element = atom.element.strip() if atom.element else atom_name[0]

        out += (
            f"ATOM  {atom_serial:5d} "
            f"{atom_name:<4}"
            f"{resname:>3} "
            f"{chain_id}"
            f"{resseq:4d}    "
            f"{x:8.3f}{y:8.3f}{z:8.3f}"
            f"  1.00 20.00           {element:>2}\n"
        )
        atom_serial += 1

    out += "END\n"

    with open(filepath, "w") as f:
        f.write(out)
# ==============================================================================
# 6. VISOR 3D TERMOFÍSICO
# ==============================================================================

def render_view(pdb, stress=None, style="cartoon"):
    pdb_clean = pdb.replace("`","")
    stress_js = json.dumps(stress.tolist()) if stress is not None else "null"

    html = f"""
    <html>
    <script src="https://3Dmol.org/build/3Dmol-min.js"></script>
    <div id="v" style="width:100%;height:500px;"></div>
    <script>
    let pdb = `{pdb_clean}`;
    let stress = {stress_js};
    let v = $3Dmol.createViewer("v",{{backgroundColor:"black"}});
    v.addModel(pdb,"pdb");

    if(stress){{
        v.setStyle({{}},{{cartoon:{{colorscheme:{{prop:'b',gradient:'roygb',min:0,max:Math.max(...stress)}}}}}});
    }} else {{
        v.setStyle({{}},{{{style}:{{color:'spectrum'}}}});
    }}

    v.zoomTo(); v.render();
    </script>
    </html>
    """
    components.html(html, height=520)
# =========================
# BLOQUE 6: STREAMLIT
# =========================

pressure = st.slider("Presión", 0.0, 10.0, 1.0)
tension = st.slider("Tensión", -5.0, 5.0, 0.0)
temperature = st.slider("Temperatura (K)", 100, 600, 300)

thermo = ThermoState(pressure, tension, temperature)
coords = apply_thermo_to_structure(structure, thermo)

if st.button("Exportar PDB"):
    export_pdb(structure, coords, "output_multisustrato.pdb")
# ==============================================================================
# 7. INTERFAZ STREAMLIT
# ==============================================================================

st.markdown("## ⚛️ OMNIKERNEL CENTURION v101.5")
st.caption("Motor Ontológico · Termodinámica Paramétrica · Exportación Multisustrato")

uploaded = st.file_uploader("Cargar PDB", type=["pdb"])
pdb_text = None

if uploaded:
    pdb_text = uploaded.getvalue().decode()

if pdb_text:
    engine = OntologicalEngine()
    structure = engine.parse_structure(pdb_text)
    q = engine.q_factor(structure)

    st.metric("Q-Factor", f"{q:.2f} uTCDS")

    # ---- CONTROLES TERMODINÁMICOS ----
    sigma = st.slider("Presión σ", 0.0, 150.0, 1.0)
    q_ext = st.slider("Q externo", -50.0, 200.0, 10.0)

    thermo = TCDS_ThermoEngine(structure)
    new_coords = thermo.apply_pressure(sigma, q_ext)
    stress = thermo.stress_field(new_coords)

    # ---- VISUALIZACIÓN ----
    render_view(pdb_text, stress=stress)

# =========================
# EXPORTACIÓN PDB CANÓNICA
# =========================

def export_pdb(structure, coords, filepath):
    """
    Exporta una estructura BioPython + coordenadas modificadas a un PDB válido.
    
    Parameters
    ----------
    structure : Bio.PDB.Structure.Structure
        Estructura base cargada (BioPython)
    coords : iterable of (x, y, z)
        Coordenadas finales (mismo orden que structure.get_atoms())
    filepath : str
        Ruta de salida .pdb
    """

    out = ""
    atom_serial = 1

    for atom, (x, y, z) in zip(structure.get_atoms(), coords):
        res = atom.get_parent()          # Residue
        chain = res.get_parent()         # Chain

        atom_name = atom.get_name()
        resname = res.get_resname()
        chain_id = chain.id if chain.id.strip() else "A"
        resseq = res.id[1]

        # Elemento químico (fallback seguro)
        element = atom.element.strip() if atom.element else atom_name[0]

        # Línea ATOM PDB estándar (80 columnas)
        out += (
            f"ATOM  {atom_serial:5d} "
            f"{atom_name:<4}"
            f"{resname:>3} "
            f"{chain_id}"
            f"{resseq:4d}    "
            f"{x:8.3f}{y:8.3f}{z:8.3f}"
            f"  1.00 20.00           {element:>2}\n"
        )

        atom_serial += 1

    out += "END\n"

    with open(filepath, "w") as f:
        f.write(out)
