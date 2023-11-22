//script.js

document.getElementById("predictionForm").addEventListener("submit", function(event) {
    event.preventDefault();

    // Collect all necessary form inputs
    let formData = {
        departureTime: document.getElementById("departureTime").value,
        weatherCondition: document.getElementById("weatherCondition").value,
        crewOnStandby: document.getElementById("crewOnStandby").value,
        aircraftType: document.getElementById("aircraftType").value,
        destinationAirport: document.getElementById("destinationAirport").value,
        departureAirport: document.getElementById("departureAirport").value,
        departureCityCode: document.getElementById("departureCityCode").value,
        fullRouting: document.getElementById("fullRouting").value,
        generalEquipmentType: document.getElementById("generalEquipmentType").value,
        elapsedJourneyTime: document.getElementById("elapsedJourneyTime").value,
        scheduleFingerprint: document.getElementById("scheduleFingerprint").value,
        passengerClasses: document.getElementById("passengerClasses").value,
        // ... continue for all other fields
    };

    let url = "http://localhost:5000/predict";

    fetch(url, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(formData)
    })
    .then(response => {
        if (response.ok) {
            return response.json();
        } else {
            throw new Error('Server response was not OK');
        }
    })
    .then(data => {
        document.getElementById("predictionResult").innerHTML = "Prediction: " + data.prediction;
    })
    .catch((error) => {
        console.error('Error:', error);
        document.getElementById("predictionResult").innerHTML = "Error in prediction: " + error.message;
    });
});

