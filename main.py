
import json
import os
from pathlib import Path
from huggingface_hub import InferenceClient

# JSON file ka path
DATA_FILE = Path(__file__).parent / "students.json"


def load_data():
    """Load student data from JSON file."""
    with open(DATA_FILE, "r", encoding="utf-8") as file:
        return json.load(file)


def save_data(data):
    """Save updated student data to JSON file."""
    with open(DATA_FILE, "w", encoding="utf-8") as file:
        json.dump(data, file, indent=2)


def find_student(data, roll_no):
    """Find a student using their roll number."""
    for student in data["students"]:
        if student["roll_no"] == str(roll_no):
            return student

    raise ValueError(f"Student with roll number {roll_no} not found.")


def calculate_attendance(roll_no):
    """Calculate total days, present days and attendance percentage."""
    data = load_data()
    student = find_student(data, roll_no)

    total_days = len(data["working_days"])
    attendance = student["attendance"]

    present_days = sum(
        1 for day in data["working_days"]
        if attendance.get(day) == "Present"
    )

    marked_days = sum(
        1 for day in data["working_days"]
        if attendance.get(day) in ("Present", "Absent")
    )

    percentage = (
        (present_days / marked_days) * 100
        if marked_days > 0 else 0
    )

    return {
        "name": student["name"],
        "roll_no": student["roll_no"],
        "total_working_days": total_days,
        "marked_days": marked_days,
        "present_days": present_days,
        "absent_days": marked_days - present_days,
        "attendance_percentage": round(percentage, 2)
    }


def calculate_attendance_marks(roll_no, max_marks=10):
    """Calculate attendance marks proportionally."""
    result = calculate_attendance(roll_no)

    marks = (
        result["attendance_percentage"] / 100
    ) * max_marks

    result["maximum_marks"] = max_marks
    result["attendance_marks"] = round(marks, 2)

    return result


def update_attendance(roll_no, day, status):
    """Update a student's attendance for a particular day."""
    data = load_data()
    student = find_student(data, roll_no)

    if day not in data["working_days"]:
        raise ValueError(f"{day} is not a valid working day.")

    if status not in ("Present", "Absent"):
        raise ValueError("Status must be Present or Absent.")

    student["attendance"][day] = status
    save_data(data)

    return {
        "message": "Attendance updated successfully.",
        "name": student["name"],
        "roll_no": student["roll_no"],
        "day": day,
        "status": status
    }


def remove_attendance(roll_no, day):
    """Remove a student's attendance record for one day."""
    data = load_data()
    student = find_student(data, roll_no)

    if day not in data["working_days"]:
        raise ValueError(f"{day} is not a valid working day.")

    if day not in student["attendance"]:
        raise ValueError("No attendance record exists for this day.")

    del student["attendance"][day]
    save_data(data)

    return {
        "message": "Attendance record removed.",
        "name": student["name"],
        "roll_no": student["roll_no"],
        "day": day,
        "status": "Unmarked"
    }


def calculate_class_average():
    """Calculate the average attendance percentage of the class."""
    data = load_data()

    results = [
        calculate_attendance(student["roll_no"])
        for student in data["students"]
    ]

    if not results:
        return {"student_count": 0, "class_average_percentage": 0}

    average = sum(
        result["attendance_percentage"] for result in results
    ) / len(results)

    return {
        "student_count": len(results),
        "class_average_percentage": round(average, 2),
        "student_results": results
    }

def ask_ai(user_instruction):
    client = InferenceClient(
        token=os.getenv("HF_TOKEN"),
        provider="auto"
    )

    response = client.chat.completions.create(
        model="Qwen/Qwen2.5-72B-Instruct",
        messages=[
            {
                "role": "system",
                "content": """
You are an attendance assistant.
Return only JSON.

Available tools:
- calculate_attendance: roll_no
- calculate_attendance_marks: roll_no, max_marks
- update_attendance: roll_no, day, status
- remove_attendance: roll_no, day
- calculate_class_average: no arguments

Examples:
{"tool":"calculate_attendance","roll_no":"101"}
{"tool":"calculate_attendance_marks","roll_no":"101","max_marks":10}
{"tool":"update_attendance","roll_no":"101","day":"Monday","status":"Absent"}
{"tool":"remove_attendance","roll_no":"101","day":"Monday"}
{"tool":"calculate_class_average"}

If information is missing, return:
{"error":"Missing information"}
"""
            },
            {"role": "user", "content": user_instruction}
        ],
        temperature=0,
        max_tokens=150
    )

    raw = response.choices[0].message.content
    command = json.loads(raw)

    if not isinstance(command, dict):
        raise ValueError("Invalid AI response")

    return command


def execute_instruction(instruction):
    command = ask_ai(instruction)

    if "error" in command:
        return command

    tool = command.get("tool")
    roll_no = command.get("roll_no")

    if tool == "calculate_attendance" and set(command) == {"tool", "roll_no"}:
        result = calculate_attendance(roll_no)

    elif tool == "calculate_attendance_marks" and set(command).issubset({"tool", "roll_no", "max_marks"}) and "roll_no" in command:
        result = calculate_attendance_marks(
            roll_no, command.get("max_marks", 10)
        )

    elif tool == "update_attendance" and set(command) == {"tool", "roll_no", "day", "status"}:
        result = update_attendance(
            roll_no, command["day"], command["status"]
        )

    elif tool == "remove_attendance" and set(command) == {"tool", "roll_no", "day"}:
        result = remove_attendance(roll_no, command["day"])

    elif tool == "calculate_class_average" and set(command) == {"tool"}:
        result = calculate_class_average()

    else:
        raise ValueError("Invalid tool or arguments from AI")

    return {"command": command, "result": result}


if __name__ == "__main__":
    print("AI Attendance System")
    print("Example: Show attendance of roll number 101")
    print("Type exit to quit.")

    while True:
        user_input = input("\nEnter instruction: ").strip()

        if user_input.lower() == "exit":
            break

        if not user_input:
            continue

        try:
            output = execute_instruction(user_input)
            print(json.dumps(output, indent=2))
        except Exception as error:
            print("Error:", error)
