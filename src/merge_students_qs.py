from __future__ import annotations
import pandas as pd
import numpy as np

DEGREE_KEYWORDS_MAP = {
    "Computer Science": [
        "computer science", "computing", "informatics",
        "software engineering", "software", "information technology",
        "information systems", "cyber", "cybersecurity",
        "artificial intelligence", "machine learning", "deep learning",
        "computer engineering", "computer systems",
        "web development", "web design", "computer animation", 
        "artificial intellience", "artificial intell", "comptuter",  
    ],

    "Data Science": [
        "data science", "data analytics", "data analysis",
        "business analytics", "analytics",
        "ciencias de datos", "ciencia de datos",
        "data and analytics"
    ],

    "Engineering - Mechanical": [
        "mechanical engineering", "automotive",
        "aerospace", "aeronautics", "astronautics"
    ],
    "Engineering - Electrical": [
        "electrical engineering", "electrical", "electronic engineering",
        "electronics", "mechatronics", "robotics",
        "telecommunications", "telecom"
    ],
    "Engineering - Chemical": [
        "chemical engineering", "biochemical engineering",
        "bioprocess", "chemical"
    ],
    "Engineering - Civil": ["civil engineering", "civil"],
    "Engineering - Mineral": ["mineral engineering"],
    "Petroleum Engineering": ["petroleum engineering"],
    "Environmental Sciences": ["environmental science", "environmental engineering"],
    "Materials Science": ["materials science"],
    "Earth & Marine Sciences": ["earth", "marine science", "ocean", "marine biology"],

    "Engineering & Technology": [
        "engineering", "technology", "engineering technology", "advanced technology"
    ],

    "Business": [
        "business", "management", "marketing", "finance", "accounting",
        "business administration", "business management",
        "international business", "entrepreneurship",
        "logistics", "supply chain", "transportation",
        "empresa", "ade", "administración y dirección de empresas",
        "bba", "iba", "dg ba", "iap"
    ],

    "Economics & Econometrics": ["economics", "econometrics", "economia"],

    "Psychology": [
        "psychology", "psychological",
        "neuroscience", "neuropsychology",
        "cognitive science", "cognitive neuroscience",
        "human neuroscience", "brain", "behaviour", "behavior"
    ],

    "Politics": [
        "politics", "political",
        "international relations", "international affairs",
        "geopolitics", "relaciones internacionales"
    ],

    "Sociology": ["sociology", "criminology", "social sciences"],
    "Social Policy": [
        "social policy", "social work", "refugees", "migration",
        "public health", "health services administration"
    ],

    "Biological": [
        "biology", "biological", "biomedical",
        "genetic", "genomics",
        "biotechnology", "microbiology",
        "molecular", "cellular", "physiology"
    ],

    "Medicine": ["medicine", "medical", "medical sciences", "biomedicine"],
    "Nursing": ["nursing", "nursing science"],
    "Dentistry": ["dentistry", "oral", "dental", "odontology"],
    "Veterinary Science": ["veterinary", "equine", "animal science"],

    "Law": ["law", "legal", "derecho"],
    "Architecture": ["architecture", "architectural"],
    "Music": ["music", "guitar", "music technology", "music production"],
    "Performing Arts": ["acting", "drama", "film", "screenwriting", "screen", "television", "radio"],
    "Arts & Humanities": [
        "liberal arts", "arts & sciences", "bachelor of arts",
        "creative writing", "humanities"
    ],
    "Natural Sciences": ["science program", "sciences", "bachelor of science", "life sciences"],
    "Mathematics": ["mathematics", "math", "algebra"],
    "Physics": ["physics", "astronomy", "theoretical physics"],
    "Chemistry": ["chemistry", "biochemistry"],
    "Sports-related Subjects": ["sports science", "exercise", "sport"],

    # Niche categories
    "Geophysics": ["geophysics"],
    "Geology": ["geology"],
    "Geography": ["geography"],
    "Statistics": ["statistics", "statistical"],
    "History of Art": ["history of art"],
    "History": ["history", "classical studies"],
    "Philosophy": ["philosophy"],
    "Theology": ["theology"],
    "Linguistics": ["linguistics"],
    "Modern Languages": ["modern languages"],
    "English Language": ["english language"],
    "Classics": ["classics"],
    "Archaeology": ["archaeology"],
    "Agriculture": ["agriculture", "agricultural"],
    "Art & Design": ["art & design", "diseño de producto"],
}

UNIVERDSITY_KEYWORDS_MAP = {
        "THE UNIVERSITY OF EDINBURG": "THE UNIVERSITY OF EDINBURGH",
        "ST. LOUIS UNIVERSITY": "SAINT LOUIS UNIVERSITY",
        "UNIVERSITY OF EAST ANGLIA (UEA)": "UNIVERSITY OF EAST ANGLIA",
        "LOUGHBOROUGH UNIV": "LOUGHBOROUGH UNIVERSITY",
        "INTO CITY, UNIVERSITY OF LONDON": "CITY, UNIVERSITY OF LONDON",
        "THE PENNSYLVANIA STATE UNIVERSITY" : "PENNSYLVANIA STATE UNIVERSITY",
        "INDIANA UNIVERSITY (BLOOMINGTON)": "INDIANA UNIVERSITY BLOOMINGTON",
        "LOYOLA UNIVERSITY OF CHICAGO": "LOYOLA UNIVERSITY CHICAGO",
        "THE UNIVERSITY OF TEXAS AT AUSTIN": "UNIVERSITY OF TEXAS AT AUSTIN",
        "UNIVERSITY OF WASHINGTON - SEATTLE":  "UNIVERSITY OF WASHINGTON",
        "CALIFORNIA INSTITUTE OF TECHNOLOGY": "CALIFORNIA INSTITUTE OF TECHNOLOGY (CALTECH)",
        "UNIVERSITY OF MINNESOTA - TWIN CITIES": "UNIVERSITY OF MINNESOTA TWIN CITIES",
        "RUTGERS UNIVERSITY (NEW BRUNCSWICK)": "RUTGERS UNIVERSITY–NEW BRUNSWICK",
        "UNIVERSITY OF MICHIGAN - ANN ARBOR": "UNIVERSITY OF MICHIGAN-ANN ARBOR",
        "UNIVERSITY OF NEW SOUTH WALES": "UNIVERSITY OF NEW SOUTH WALES (UNSW SYDNEY)",
        "MASSACHUSETTS INSTITUTE OF TECHNOLOGY": "MASSACHUSETTS INSTITUTE OF TECHNOLOGY (MIT)",
        "UNIVERSITY OF MARYLAND - COLLEGE PARK": "UNIVERSITY OF MARYLAND COLLEGE PARK",
        "UNIVERSITY OF NORTH CAROLINA - CHAPEL HILL": "UNIVERSITY OF NORTH CAROLINA CHAPEL HILL",
        "UNIVERSITY OF ILLINOIS - URBANA CHAMPAIGN": "UNIVERSITY OF ILLINOIS AT URBANA-CHAMPAIGN",
        "UNIVERSITY OF ST. ANDREWS": "UNIVERSITY OF ST ANDREWS",
        "KING'S COLLEGE OF LONDON": "KING'S COLLEGE LONDON",
        "UNIVERSITY OF EDINBURG": "UNIVERSITY OF EDINBURGH",  
        "IE UNIVERSITY SEGOVIA": "IE UNIVERSITY",
        "IE UNIVERSITY MADRID CAMPUS": "IE UNIVERISTY"
        }


def university_mapping(university: str, uni_map: dict[str, str] = UNIVERDSITY_KEYWORDS_MAP) -> str:
    university = university.upper().strip()
    university = university.replace("&", "AND")
    university = university.replace(" OF ", " OF ")  # conserva estructura
    university = university.replace("U.", "UNIVERSITY")
    university = university.replace("U ", "UNIVERSITY ")  # cuidado con "U" sola
    university = university.replace("THE ", "")
    university = university.replace("UNIVERSIDAD", "UNIVERSITY")
    university = university.replace(" DE ", " OF ")
    university = university.replace(", ", " ")
    university = university.replace("INTO ", "")
    university = university.replace("DEUSTU", "DEUSTO")


    if university in uni_map:
        return uni_map[university]
    return university


# Privada
# Categoriza los grados de fl segun los grados de qs, basandose en las 
# palabras clave de DEGREE_KEYWORDS_MAP, pero permitiendo que un grado 
# pueda pertenecer a varias categorias (ej: "Computer Science and Business" -> ["Computer Science", "Business"])
def _categorizar_degree_multiple(degree, degree_map: dict[str, list[str]] = DEGREE_KEYWORDS_MAP) -> list[str]:
    degree_lower = degree.lower()
    categorias = []
    for category, keywords in degree_map.items():
        if any(keyword in degree_lower for keyword in keywords):
            categorias.append(category)
    return categorias if categorias else ["Uncategorized"]



def degree_mapping(df_students: pd.DataFrame, degree_map: dict[str, list[str]] = DEGREE_KEYWORDS_MAP) -> tuple[pd.DataFrame, dict[str, list[str]]]:
    df = df_students.copy()

    df_students_degrees = df["degree"].unique()
    df_qs_degrees = {cat: [] for cat in degree_map.keys()}

    # Asignación
    for degree in df_students_degrees:
        degree_lower = degree.lower()
        matched = False
        for category, keywords in degree_map.items():
            if any(keyword in degree_lower for keyword in keywords):
                df_qs_degrees[category].append(degree)
                matched = True
                break
        if not matched:
            df_qs_degrees.setdefault('Uncategorized', []).append(degree)

    # Aplicar la función y expandir filas
    df["degree"] = df["degree"].apply(lambda x: _categorizar_degree_multiple(x, degree_map=degree_map))
    df = df.explode("degree").reset_index(drop=True)
    
    return df, df_qs_degrees


# Mostrar la asignacion de carreras de df_students a categorias de df_qs
def show_degree_mapping(df_qs_degrees: dict[str, list[str]]) -> None:
    for cat, items in df_qs_degrees.items():
        if items:
            print(f"\n{cat} ({len(items)}):")
            for item in items:
                print(f"  - {item}")


def show_inner_universities(df_students: pd.DataFrame, df_qs: pd.DataFrame) -> None:
    inner = set(df_students["university"].dropna().unique()) & set(df_qs["university"].dropna().unique())
    print(f"Universidades únicas students: {len(df_students['university'].dropna().unique())}")
    print(f"Universidades únicas qs: {len(df_qs['university'].dropna().unique())}")
    print(f"Universidades en común: {len(inner)}")


def merge_students_qs(df_students: pd.DataFrame, df_qs: pd.DataFrame) -> pd.DataFrame:
    df_students = df_students.copy()
    df_qs = df_qs.copy()

    df = pd.merge(df_students, df_qs, on=["university", "degree"], how="left")
    df = df.drop(columns=["country_y"], errors='ignore').rename(columns={"country_x": "country"})

    return df