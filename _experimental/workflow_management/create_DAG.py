from airflow import DAG
from airflow.operators.bash_operator import BashOperator
from datetime import datetime, timedelta

# Define the DAG
default_args = {
    'owner': 'your_name',
    'depends_on_past': False,
    'start_date': datetime(2023, 4, 15),
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

dag = DAG(
    'two_branches_dag',
    default_args=default_args,
    description='A DAG with two branches of tasks',
    schedule_interval=timedelta(days=1),
)

# Define the tasks for branch 1
task1 = BashOperator(
    task_id='task1',
    bash_command='echo "Task 1"',
    dag=dag,
)

task2 = BashOperator(
    task_id='task2',
    bash_command='echo "Task 2"',
    dag=dag,
)

task3 = BashOperator(
    task_id='task3',
    bash_command='echo "Task 3"',
    dag=dag,
)

# Define the tasks for branch 2
task4 = BashOperator(
    task_id='task4',
    bash_command='echo "Task 4"',
    dag=dag,
)

task5 = BashOperator(
    task_id='task5',
    bash_command='echo "Task 5"',
    dag=dag,
)

# Define the task dependencies for the first branch
task1 >> task2 >> task3

# Define the task dependencies for the second branch
task1 >> task4 >> task5
