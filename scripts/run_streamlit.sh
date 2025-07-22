# Need to port forwarding to the EC2 instance to access Qdrant
# Run this command in terminal:
# aws ssm start-session \
#   --target i-058c5440d1c3ad6c9 \
#   --document-name AWS-StartPortForwardingSession \
#   --parameters '{"portNumber":["6333"], "localPortNumber":["6333"]}'

echo "Starting Streamlit app..."
streamlit run app/main.py

# need to make it executable
# chmod +x scripts/run_streamlit.sh
# then run it with ./scripts/run_streamlit.sh