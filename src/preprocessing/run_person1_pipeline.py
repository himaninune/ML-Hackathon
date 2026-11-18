from src.preprocessing.audit_dataset import main as audit
from src.preprocessing.prepare_dataset import main as prepare
from src.preprocessing.build_network import main as network
from src.candidate_generation.generate_candidates import main as candidates
from src.features.build_features import main as features
from src.preprocessing.make_visualizations import main as maps

def main():
    print("\n[1/6] Dataset audit")
    audit()
    print("\n[2/6] GPS preprocessing")
    prepare()
    print("\n[3/6] Road-network construction")
    network()
    print("\n[4/6] Candidate generation")
    candidates()
    print("\n[5/6] Feature export")
    features()
    print("\n[6/6] Folium visualizations")
    maps()
    print("\nPERSON 1 PIPELINE FINISHED.")

if __name__ == "__main__":
    main()
