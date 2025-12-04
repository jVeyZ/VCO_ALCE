import os
import json
import pandas as pd # type: ignore
from pathlib import Path
from datetime import datetime

def load_answer_template(template_path):
    """Load the answer template (blueprint) from frontend/answer_key"""
    try:
        with open(template_path, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Error: Template not found at {template_path}")
        return None

def extract_correct_answers(blueprint):
    """Extract correct answers from blueprint"""
    correct_answers = {}
    
    if not blueprint or 'questionBank' not in blueprint:
        return correct_answers
    
    for idx, question in enumerate(blueprint['questionBank'], 1):
        if question['type'] == 'mcq':
            correct_answers[str(idx)] = question.get('correctOption', 'N/A')
        elif question['type'] == 'numeric':
            correct_answers[str(idx)] = str(question.get('numericAnswer', 'N/A'))
    
    return correct_answers

def load_student_answers(json_file_path):
    """Load student answers from processed JSON"""
    try:
        with open(json_file_path, 'r') as f:
            data = json.load(f)
            return {
                'dni': data.get('dni', 'N/A'),
                'answers': data.get('answers', {}),
                'image': data.get('image', 'N/A')
            }
    except FileNotFoundError:
        print(f"Error: JSON file not found at {json_file_path}")
        return None
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON in {json_file_path}")
        return None

def compare_answers(correct_answers, student_answers):
    """Compare student answers with correct answers"""
    results = {
        'correct': [],
        'incorrect': [],
        'missing': []
    }
    
    for question_id in sorted(correct_answers.keys(), key=lambda x: int(x)):
        correct = correct_answers[question_id]
        student = student_answers.get(question_id, None)
        
        # Check if answer is missing or empty
        if student is None or str(student).strip() == '':
            results['missing'].append(question_id)
        elif str(student).strip().upper() == str(correct).strip().upper():
            results['correct'].append(question_id)
        else:
            results['incorrect'].append(question_id)
    
    return results

def generate_report(results_list, output_dir):
    """Generate Excel report from results"""
    
    report_data = []
    
    for result in results_list:
        dni = result['dni']
        comparisons = result['comparisons']
        
        total_questions = len(comparisons['correct']) + len(comparisons['incorrect']) + len(comparisons['missing'])
        correct_count = len(comparisons['correct'])
        incorrect_count = len(comparisons['incorrect'])
        missing_count = len(comparisons['missing'])
        score = (correct_count / total_questions * 100) if total_questions > 0 else 0
        
        report_data.append({
            'DNI': dni,
            'Correctas': correct_count,
            'Incorrectas': incorrect_count,
            'Sin Responder': missing_count,
            'Total Preguntas': total_questions,
            'Puntuación (%)': round(score, 2),
            'Preguntas Correctas': ','.join(comparisons['correct']) if comparisons['correct'] else '-',
            'Preguntas Incorrectas': ','.join(comparisons['incorrect']) if comparisons['incorrect'] else '-',
            'Preguntas Sin Responder': ','.join(comparisons['missing']) if comparisons['missing'] else '-'
        })
    
    df = pd.DataFrame(report_data)
    
    # Generate filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(output_dir, f"correcciones_{timestamp}.xlsx")
    
    # Create Excel with formatting
    with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
        df.to_excel(writer, sheet_name='Resultados', index=False)
        
        # Get the workbook and worksheet
        workbook = writer.book
        worksheet = writer.sheets['Resultados']
        
        # Auto-adjust column widths
        for column in worksheet.columns:
            max_length = 0
            column_letter = column[0].column_letter
            for cell in column:
                try:
                    if len(str(cell.value)) > max_length:
                        max_length = len(str(cell.value))
                except:
                    pass
            adjusted_width = min(max_length + 2, 50)
            worksheet.column_dimensions[column_letter].width = adjusted_width
    
    return output_file

def main():
    # Paths
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    json_test_dir = project_root / 'img' /'json_test'
    output_dir = project_root / 'output'
    answer_key_dir = project_root / 'frontend' / 'answer_key'
    
    # Create output directory if it doesn't exist
    output_dir.mkdir(exist_ok=True)
    
    # Find the answer template (most recent or specified)
    if not answer_key_dir.exists():
        print(f"Error: Answer key directory not found at {answer_key_dir}")
        return
    
    answer_keys = list(answer_key_dir.glob('*.json'))
    if not answer_keys:
        print(f"Error: No answer key JSON files found in {answer_key_dir}")
        return
    
    # Use the most recently modified answer key
    template_path = max(answer_keys, key=lambda p: p.stat().st_mtime)
    print(f"Using answer template: {template_path.name}")
    
    blueprint = load_answer_template(template_path)
    if not blueprint:
        return
    
    correct_answers = extract_correct_answers(blueprint)
    print(f"Loaded {len(correct_answers)} correct answers from template")
    
    # Process all student answer JSONs
    if not json_test_dir.exists():
        print(f"Error: json_test directory not found at {json_test_dir}")
        return
    
    json_files = list(json_test_dir.glob('*_answers.json'))
    if not json_files:
        print(f"Error: No answer JSON files found in {json_test_dir}")
        return
    
    print(f"Found {len(json_files)} student answer files to process")
    
    results_list = []
    
    for json_file in json_files:
        print(f"\nProcessing: {json_file.name}")
        
        student_data = load_student_answers(json_file)
        if not student_data:
            continue
        
        comparisons = compare_answers(correct_answers, student_data['answers'])
        
        # Print summary
        correct_count = len(comparisons['correct'])
        incorrect_count = len(comparisons['incorrect'])
        missing_count = len(comparisons['missing'])
        total = correct_count + incorrect_count + missing_count
        score = (correct_count / total * 100) if total > 0 else 0
        
        print(f"  DNI: {student_data['dni']}")
        print(f"  Correctas: {correct_count}, Incorrectas: {incorrect_count}, Sin responder: {missing_count}")
        print(f"  Puntuación: {score:.2f}%")
        
        results_list.append({
            'dni': student_data['dni'],
            'comparisons': comparisons
        })
    
    if not results_list:
        print("\nNo student data was processed.")
        return
    
    # Generate Excel report
    print(f"\nGenerating Excel report...")
    report_file = generate_report(results_list, output_dir)
    print(f"Report saved to: {report_file}")

if __name__ == '__main__':
    main()
