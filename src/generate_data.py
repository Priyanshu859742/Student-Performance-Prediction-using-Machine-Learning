import pandas as pd
import numpy as np
import os

def generate_student_data(num_samples=1000):
    np.random.seed(42)
    
    # Generate features
    study_hours = np.random.uniform(1, 10, num_samples) # Hours per day
    attendance = np.random.uniform(50, 100, num_samples) # Percentage
    previous_marks = np.random.uniform(40, 100, num_samples) # Percentage
    assignments_completed = np.random.randint(0, 11, num_samples) # 0 to 10
    
    # Generate Target: Final Score based on features + some noise
    # Base score
    final_score = (
        0.3 * previous_marks + 
        2.5 * study_hours + 
        0.2 * attendance + 
        2.0 * assignments_completed + 
        np.random.normal(0, 5, num_samples) # Noise
    )
    
    # Clip final score to be within 0 to 100
    final_score = np.clip(final_score, 0, 100)
    
    # Generate Target: Pass/Fail classification
    # Pass if final score >= 50
    pass_fail = np.where(final_score >= 50, 1, 0)
    
    # Create DataFrame
    df = pd.DataFrame({
        'Study_Hours': study_hours.round(2),
        'Attendance_Percentage': attendance.round(2),
        'Previous_Marks': previous_marks.round(2),
        'Assignments_Completed': assignments_completed,
        'Final_Score': final_score.round(2),
        'Pass_Fail': pass_fail
    })
    
    return df

if __name__ == "__main__":
    print("Generating synthetic student performance dataset...")
    df = generate_student_data(1000)
    
    # Define output path
    output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'student_performance.csv')
    
    df.to_csv(output_path, index=False)
    print(f"Dataset generated and saved to: {output_path}")
    print("First 5 rows:")
    print(df.head())
