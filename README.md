# **Gesture Recognition Using AI**

A real-time gesture recognition system built using Python, OpenCV, and MediaPipe. 
This project enables sign language recognition, translating gestures into text and saving the recognized text to a file. 
Designed to assist individuals with hearing or speech impairments by bridging the communication gap.


## **Features**

- **Real-Time Gesture Recognition:** Uses a webcam to detect hand gestures.
- **Sign Language Translation:** Converts gestures into alphabets and displays the result.
- **Save Recognized Text:** Saves the recognized gestures as text into a file.
- **User-Friendly GUI:** Includes a Tkinter-based graphical interface for easy interaction.
- **Scalable Design:** Supports easy addition of new gestures or sign language alphabets.


## **Technologies Used**

- **Programming Language:** Python
- **Libraries/Frameworks:**
  - OpenCV: For webcam video capture and image processing.
  - MediaPipe: For hand landmarks and gesture tracking.
  - Tkinter: For building the graphical user interface.
  - Scikit-learn: For gesture classification using machine learning.


## **Project Structure**

```
Gesture-Recognition/
│
├── gesture_recognition.py       # Main script for gesture detection and recognition
├── gui.py                       # Tkinter-based GUI integration script
├── model.p                      # Pre-trained Random Forest model
├── README.md                    # Documentation 
├── requirements.txt             # List of dependencies
└── assets/                      # Folder for images.
```

## **Installation**

### **Prerequisites**
1. Python 3.10 or later
2. A working webcam

### **Steps**
1. Clone the repository:
   ```bash
   git clone https://github.com/Griffin2005/Gesture-Recognition.git
   cd Gesture-Recognition
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Test your webcam setup:
   ```bash
   python gesture_recognition.py
   ```

---

## **Usage**

### **Run the Gesture Recognition**
Execute the following command to start the application:
```bash
python gesture_recognition.py
```

### **Run the GUI**
Execute the following command to launch the GUI:
```bash
python gui.py
```

### **Functionalities**
1. **Gesture Detection:** Use gestures visible to the webcam.
2. **View Recognized Gestures:** Results are displayed in real-time on the GUI text box.
3. **Clear Text:** Click "Clear All" to reset the recognized text.
4. **Save Recognized Text:** Save the recognized gestures to a text file for later use.


## **How It Works**

1. **Hand Detection:**  
   - MediaPipe identifies hand landmarks in the webcam feed.
   - The coordinates of these landmarks are used as input features.

2. **Data Collection**  
   - The `Data Collection.py` script collects the data in form of images for the alphabets (Total 26 folders, each representing each alphabet and each folder contains 100 images).

3. **Dataset Creation:**
   - The `Dataset creation.py` script maps the landmarks on the hand and creates a dataset for each alphabet.
  
4. **Train classifier**
   - The `Train classifier.py` script trains a Random Forect Model and displays the accuracy score.

3. **Gesture Classification:**  
   - A pre-trained Random Forest model (`model.p`) predicts the gesture based on features.

4. **Integration with GUI:**  
   - The recognized gestures are displayed and saved via a Tkinter-based GUI.

---

## **Customization**

- **Adding New Gestures:**  
  Updating the training dataset with new gestures and retrain the model.
  
- **Model Training:**  
  Replacing `model.p` with a new model trained using scikit-learn or any ML framework.

- **Enhancing GUI:**  
  Modifying `gui.py` to include new features like live video feed or audio output.

---

## **Contributing**

Contributions are welcome! Feel free to fork the repository and submit a pull request. 

---

## **License**

This project is licensed under the MIT License. See the `LICENSE` file for details.

---

## **Contact**

For any questions or feedback, feel free to reach out:

- **Email:** harshmaurya0509@gmail.com
- **GitHub:** [Griffin2005](https://github.com/Griffin2005)
