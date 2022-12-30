import unittest
from airflow.models import DagBag

class TestDags(unittest.TestCase):
    def test_dag_imports(self):
        dag_bag = DagBag(dag_folder="../dags/", include_examples=False)
        assert dag_bag.dags is not None
        assert len(dag_bag.import_errors) == 0

    def test_dag_01_loaded(self):
        dag_bag = DagBag(dag_folder="../dags/", include_examples=False)
        dag = dag_bag.dags['01_generate_dataset']
        assert "01_generate_dataset" in dag_bag.dags
        assert  len(dag_bag.import_errors) == 0
        assert len(dag_bag.dags["01_generate_dataset"].tasks) == 1
        assert dag is not None

    def test_dag_02_loaded(self):
        dag_bag = DagBag(dag_folder="../dags/", include_examples=False)
        dag = dag_bag.dags['02_weekly_pipeline']
        assert "02_weekly_pipeline" in dag_bag.dags
        assert len(dag_bag.import_errors) == 0
        assert dag is not None
        assert len(dag_bag.dags["02_weekly_pipeline"].tasks) == 6

    def test_dag_03_loaded(self):
        dag_bag = DagBag(dag_folder="../dags/", include_examples=False)
        dag = dag_bag.dags['03_daily_pipeline']
        assert "03_daily_pipeline" in dag_bag.dags
        assert len(dag_bag.import_errors) == 0
        assert dag is not None
        assert len(dag_bag.dags["03_daily_pipeline"].tasks) == 2

    def test_dag_02_structure(self):
        dag_tasks={
            "wait_for_raw_data":{"preprocessing_data"},
            "wait_for_raw_target":{"preprocessing_data"},
            "preprocessing_data":{"train_val_split"},
            "train_val_split":{"fitting_model"},
            "fitting_model":{"model_validation"},
            "model_validation":set()
        }
        dag_bag = DagBag(dag_folder="../dags/", include_examples=False)
        dag = dag_bag.dags['02_weekly_pipeline']
        for name, task in dag.task_dict.items():
            assert(task.downstream_task_ids == dag_tasks[name])

    def test_dag_03_structure(self):
        dag_tasks={
            "wait_for_model":{"pipeline"},
            "pipeline":set()
        }
        dag_bag = DagBag(dag_folder="../dags/", include_examples=False)
        dag = dag_bag.dags['03_daily_pipeline']
        for name, task in dag.task_dict.items():
            assert(task.downstream_task_ids == dag_tasks[name])

if __name__ == '__main__':
    unittest.main()