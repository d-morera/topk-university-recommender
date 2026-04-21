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
    """Normalize a university name for matching/merging.

    Notes:
        This is a deterministic string-normalization heuristic + optional overrides.

    Args:
        university: Raw university name.
        uni_map: Optional dictionary of exact-name overrides.

    Returns:
        Normalized university name.
    """
    university = university.upper().strip()
    university = university.replace("&", "AND")
    university = university.replace(" OF ", " OF ")
    university = university.replace("U.", "UNIVERSITY")
    university = university.replace("U ", "UNIVERSITY ")
    university = university.replace("THE ", "")
    university = university.replace("UNIVERSIDAD", "UNIVERSITY")
    university = university.replace(" DE ", " OF ")
    university = university.replace(", ", " ")
    university = university.replace("INTO ", "")
    university = university.replace("DEUSTU", "DEUSTO")

    if university in uni_map:
        return uni_map[university]
    return university


def _categorize_degrees(degree, degree_map: dict[str, list[str]] = DEGREE_KEYWORDS_MAP) -> list[str]:
    """Map a raw degree string to one-or-more QS categories (multi-label).

    Notes:
        Uses substring keyword matching against `DEGREE_KEYWORDS_MAP`. Some degrees can
        legitimately map to multiple categories (e.g., "Computer Science and Business").

    Args:
        degree: Raw degree/program string.
        degree_map: Category -> keyword list.

    Returns:
        List of matched categories, or ["Uncategorized"] if none match.
    """
    degree_lower = degree.lower()
    categorias = []
    for category, keywords in degree_map.items():
        if any(keyword in degree_lower for keyword in keywords):
            categorias.append(category)
    return categorias if categorias else ["Uncategorized"]



def degree_mapping(df_students: pd.DataFrame, degree_map: dict[str, list[str]] = DEGREE_KEYWORDS_MAP) -> tuple[pd.DataFrame, dict[str, list[str]]]:
    """Expand student rows by mapping `degree` into QS subject categories.

    This function returns:
      1) an expanded DataFrame where `degree` is replaced by mapped categories and
         exploded to one row per category
      2) a summary dict of category -> list of degrees seen

    Args:
        df_students: Students DataFrame with a `degree` column.
        degree_map: Mapping from QS category to keyword list.

    Returns:
        (expanded_df, mapping_summary)
    """
    df = df_students.copy()

    df_students_degrees = df["degree"].unique()
    df_qs_degrees = {cat: [] for cat in degree_map.keys()}

    # Build a quick summary: which raw degrees ended up in which category.
    for degree in df_students_degrees:
        degree_lower = degree.lower()
        matched = False
        for category, keywords in degree_map.items():
            if any(keyword in degree_lower for keyword in keywords):
                df_qs_degrees[category].append(degree)
                matched = True
                break
        if not matched:
            df_qs_degrees.setdefault("Uncategorized", []).append(degree)

    # Apply multi-label mapping and explode into one row per category.
    df["degree"] = df["degree"].apply(lambda x: _categorize_degrees(x, degree_map=degree_map))
    df = df.explode("degree").reset_index(drop=True)

    return df, df_qs_degrees


def show_degree_mapping(df_qs_degrees: dict[str, list[str]]) -> None:
    """Pretty-print the degree-to-category mapping summary."""
    for cat, items in df_qs_degrees.items():
        if items:
            print(f"\n{cat} ({len(items)}):")
            for item in items:
                print(f"  - {item}")


def show_inner_universities(df_students: pd.DataFrame, df_qs: pd.DataFrame) -> None:
    """Print basic intersection stats between student and QS university sets."""
    inner = set(df_students["university"].dropna().unique()) & set(df_qs["university"].dropna().unique())
    print(f"Unique universities in students: {len(df_students['university'].dropna().unique())}")
    print(f"Unique universities in QS: {len(df_qs['university'].dropna().unique())}")
    print(f"Universities in common: {len(inner)}")



def merge_students_qs(df_students: pd.DataFrame, df_qs: pd.DataFrame) -> pd.DataFrame:
    """Left-join students with QS scores using (university, degree) as keys.

    Notes:
        This merge assumes QS has been cleaned to use the same canonical column names.
        Country columns are harmonized by keeping `country_x` from students.

    Args:
        df_students: Cleaned/expanded students DataFrame.
        df_qs: Cleaned QS DataFrame.

    Returns:
        Merged DataFrame with `qs_score` attached where available.
    """
    df_students = df_students.copy()
    df_qs = df_qs.copy()

    df = pd.merge(df_students, df_qs, on=["university", "degree"], how="left")
    df = df.drop(columns=["country_y"], errors="ignore").rename(columns={"country_x": "country"})

    return df