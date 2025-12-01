from data_analysis_pipeline.splits.splits_main import splits_main

def run(cfg):
    print("Launching data splitting")
    splits_main(cfg)
    print("Done.")