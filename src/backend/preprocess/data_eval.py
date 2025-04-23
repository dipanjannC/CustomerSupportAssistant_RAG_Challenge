from evidently import Report
from evidently import Dataset,DataDefinition
from evidently.descriptors import Sentiment, TextLength, Contains
from evidently.presets import TextEvals

from pathlib import Path
import pandas as pd
import os


project_root = Path(__file__).resolve().parents[3]
raw_dataset = os.path.join(project_root,"data" ,"raw" , "raw_customer_support_dataset.json")

def get_raw_eval_report():
    # print(project_root)
    df = pd.read_json(raw_dataset,lines=True)

    eval_dataset = Dataset.from_pandas(pd.DataFrame(df),
    data_definition=DataDefinition(),
    descriptors=[
        Sentiment("output", alias="Sentiment"),
        TextLength("output", alias="Length"),
        Contains("output", items=['', 'apologize'], mode="any", alias="Denials")
    ])

    print(eval_dataset.as_dataframe())

    report = Report([
        TextEvals()
    ])

    data_eval = report.run(eval_dataset)
    return data_eval


if __name__ == "__main__":


    data_eval = get_raw_eval_report()
    # print(data_eval.dict())

    # Save the report to a HTML file
    data_eval.save_html(os.path.join(project_root, "data", "output", "report.html"))

    # Save the report to a JSON file
    data_eval.save_json(os.path.join(project_root, "data", "output", "report.json"))
    # Save the report to a CSV file


