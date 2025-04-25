echo "Running simulations..."

cd /home/ubuntu/EASE_2025_active_learning_LLM

for i in {1..2000}
do
    echo "Running simulation iteration $i"
    .venv/bin/python code/simulations.py 
done