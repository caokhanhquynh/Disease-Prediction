var symNum = 1;
var new_symptoms = [
    "abdominal_pain", "acute_liver_failure", "altered_sensorium", "anxiety", "back_pain",
    "belly_pain", "blackheads", "blister", "blood_in_sputum", "bloody_stool", "blurred_and_distorted_vision",
    "brittle_nails", "breathlessness", "bruising", "burning_micturition", "chest_pain", "chills", "coma",
    "congestion", "continuous_feel_of_urine", "continuous_sneezing", "cough", "cramps", "dark_urine",
    "dehydration", "depression", "diarrhoea", "dischromic _patches", "distention_of_abdomen", "dizziness",
    "drying_and_tingling_lips", "enlarged_thyroid", "excessive_hunger", "extra_marital_contacts", "fainting",
    "fast_heart_rate", "fatigue", "fluid_overload", "foul_smell_of urine", "frequent_urination",
    "headache", "high_fever", "history_of_alcohol_consumption", "inflammatory_nails", "indigestion",
    "internal_itching", "irregular_sugar_level", "irritability", "irritation_in_anus", "itching",
    "joint_pain", "knee_pain", "lack_of_concentration", "lethargy", "loss_of_appetite", "loss_of_balance",
    "loss_of_smell", "malaise", "mild_fever", "moles", "mood_swings", "movement_stiffness", "mucoid_sputum",
    "muscle_pain", "muscle_wasting", "muscle_weakness", "nausea", "neck_pain", "nodal_skin_eruptions",
    "obesity", "pain_behind_the_eyes", "pain_during_bowel_movements", "pain_in_anal_region", "painful_walking",
    "palpitations", "passage_of_gases", "patches_in_throat", "phlegm", "polyuria", "prognosis",
    "prominent_veins_on_calf", "puffy_face_and_eyes", "pus_filled_pimples", "red_sore_around_nose",
    "red_spots_over_body", "receiving_blood_transfusion", "receiving_unsterile_injections", "restlessness",
    "rusty_sputum", "runny_nose", "scurring", "shivering", "silver_like_dusting", "sinus_pressure",
    "skin_peeling", "skin_rash", "small_dents_in_nails", "spinning_movements", "spotting_ urination",
    "stiff_neck", "stomach_bleeding", "stomach_pain", "sunken_eyes", "swelling_joints", "swelling_of_stomach",
    "swelled_lymph_nodes", "swollen_blood_vessels", "swollen_extremeties", "swollen_legs",
    "throat_irritation", "toxic_look_(typhos)", "ulcers_on_tongue", "unsteadiness", "visual_disturbances",
    "vomiting", "watering_from_eyes", "weakness_in_limbs", "weakness_of_one_body_side", "weight_gain",
    "weight_loss", "yellow_crust_ooze", "yellowing_of_eyes", "yellowish_skin"
];

// Get the dropdown element
var dropdown = document.getElementById("dropdown");

function populateDropdown(dropdown) {
    // Populate the dropdown with options
    new_symptoms.forEach(function(symptom) {
        var option = document.createElement("option");
        option.value = symptom;
        option.textContent = symptom.replace(/_/g, " "); // Replace underscores with spaces for readability
        dropdown.appendChild(option);
    });
}

function addSymptoms(){
    //update number of symtoms
    symNum += 1;

    // Create a new div element for the symptom
    var newDiv = document.createElement('div');
    newDiv.classsName = 'enter_symptoms';
    
    // Create a label for the new symptom dropdown
    var newLabel = document.createElement('label');
    newLabel.textContent = 'Symptom ' + symNum;
    newLabel.htmlFor = 'symptoms';

    // Create a new select element
    var newSelect = document.createElement('select');
    newSelect.name = 'symptoms';
    newSelect.className = 'dropdown';

    // Populate the new select element
    populateDropdown(newSelect);

    // Append the label and select to the new div
    newDiv.appendChild(newLabel);
    newDiv.appendChild(newSelect);

    // Append the new div to the symptom container
    var container = document.getElementById('symptomContainer');
    container.appendChild(newDiv);

}

// Populate the initial dropdown on page load
document.addEventListener('DOMContentLoaded', function() {
    var initialDropdown = document.querySelector('.dropdown');
    populateDropdown(initialDropdown);
});