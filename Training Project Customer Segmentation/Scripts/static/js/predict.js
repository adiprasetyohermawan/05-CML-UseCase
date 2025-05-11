// Function to toggle sections
function showSection(sectionId) {
  // Hide all sections
  document.getElementById("home-section").style.display = "none";
  document.getElementById("visualization-section").style.display = "none";
  document.getElementById("prediction-section").style.display = "none";

  // Show the selected section
  document.getElementById(sectionId).style.display = "block";
}

function formatInputValue(inputElement, rawValueElementId) {
  // Get the raw numerical value without formatting
  const rawValue = inputElement.value.replace(/\D/g, "");

  // Update the hidden input with the raw value for processing
  document.getElementById(rawValueElementId).value = rawValue;

  // Format the displayed value with thousand separators
  inputElement.value = new Intl.NumberFormat("de-DE").format(rawValue);
}

// Submit prediction form
function submitPrediction() {
  const data = {
    dob: document.getElementById("dob").value,
    bank_id: document.getElementById("bank_id").value,
    city: document.getElementById("city").value,
    total_transaction: parseFloat(
      document.getElementById("totalTransactionRaw").value
    ),
    income: parseFloat(document.getElementById("incomeRaw").value),
  };

  axios
    .post("/predict", data)
    .then((response) => {
      const result = response.data;
      document.getElementById("prediction-result").innerHTML = `
        <div class="prediction-result-container">
          <p class="cluster-text">Cluster: ${result.customer_cluster}</p>
          <p class="interpretation-text">${result.interpretation}</p>
        </div>`;
      // document.getElementById(
      //   "test-result"
      // ).innerHTML = `<p>Test output: ${JSON.stringify(data)}</p>`;
    })
    .catch((error) => console.error(error));
}

async function loadClusterData() {
  const clusterId = document.getElementById("clusterSelect").value;

  try {
    const summaryResponse = await fetch(`/get_cluster_summary`);
    const histogramResponse = await fetch(`/get_histogram/${clusterId}`);

    if (!summaryResponse.ok || !histogramResponse.ok) {
      throw new Error("Failed to fetch summary or histogram data");
    }

    const summaryData = await summaryResponse.json();
    const histogramData = await histogramResponse.json();

    // console.log("Summary Data:", summaryData); // Check summary data structure
    // console.log("Histogram Data:", histogramData); // Check histogram data structure

    if (summaryData[clusterId] && histogramData.img_data) {
      displaySummaryAndHistogram(
        clusterId,
        summaryData,
        histogramData.img_data
      );
      displayClusterInterpretation(clusterId);
      // Show the "View Feature Encoding Notes" button and interpretation summary section
      document.getElementById("encodingNotesContainer").style.display = "block";
      document.getElementById("interpretationSummary").style.display = "block";
    } else {
      console.error(
        "Summary or histogram data is missing for cluster:",
        clusterId
      );
    }
  } catch (error) {
    console.error("Error fetching data:", error);
  }
}

function displaySummaryAndHistogram(
  clusterId,
  summaryData,
  histogramImageData
) {
  // Display summary table
  const summaryTableDiv = document.getElementById("summaryTable");
  summaryTableDiv.innerHTML = `<h3>Cluster ${clusterId} Table Summary</h3>`;

  let tableHtml = `
    <table class="table table-bordered">
      <thead>
        <tr>
          <th>Feature</th>
          <th>Mean</th>
          <th>Std</th>
          <th>Min</th>
          <th>25%</th>
          <th>50%</th>
          <th>75%</th>
          <th>Max</th>
        </tr>
      </thead>
      <tbody>
  `;

  const clusterSummary = summaryData[clusterId];

  if (!clusterSummary) {
    console.error("No summary data found for this cluster:", clusterId);
    return;
  }

  for (const feature in clusterSummary) {
    const featureData = clusterSummary[feature];
    tableHtml += `<tr>
      <td>${feature}</td>
      <td>${
        featureData?.mean !== undefined ? featureData.mean.toFixed(2) : "N/A"
      }</td>
      <td>${
        featureData?.std !== undefined ? featureData.std.toFixed(2) : "N/A"
      }</td>
      <td>${featureData?.min ?? "N/A"}</td>
      <td>${featureData?.["25%"] ?? "N/A"}</td>
      <td>${featureData?.["50%"] ?? "N/A"}</td>
      <td>${featureData?.["75%"] ?? "N/A"}</td>
      <td>${featureData?.max ?? "N/A"}</td>
    </tr>`;
  }

  tableHtml += "</tbody></table>";
  summaryTableDiv.innerHTML += tableHtml;

  // Display histogram image
  const histogramImageDiv = document.getElementById("histogramImage");
  histogramImageDiv.innerHTML = `
    <h3>Cluster ${clusterId} Histogram</h3>
    <img src="data:image/png;base64,${histogramImageData}" class="img-fluid" alt="Cluster ${clusterId} Histogram">
  `;
}

// Function to display the interpretation for the selected cluster
function displayClusterInterpretation(clusterId) {
  fetch("static/js/analysis-interpretation.JSON")
    .then((response) => response.json())
    .then((data) => {
      // Get the interpretation details for the selected cluster
      const interpretation = data[clusterId];

      // Update the cluster title
      document.getElementById("clusterTitle").innerText = interpretation.title;

      // Populate the sections
      const sectionsContainer = document.getElementById("clusterSections");
      sectionsContainer.innerHTML = ""; // Clear any existing sections

      interpretation.sections.forEach((section) => {
        // Create elements for each section
        const sectionTitle = document.createElement("h4");
        sectionTitle.innerText = section.subtitle;

        const sectionContent = document.createElement("p");
        sectionContent.innerText = section.content;

        // Append to the sections container
        sectionsContainer.appendChild(sectionTitle);
        sectionsContainer.appendChild(sectionContent);
      });

      // Update the summary
      document.getElementById("clusterSummary").innerText =
        interpretation.summary;
    })
    .catch((error) =>
      console.error("Error loading interpretation data:", error)
    );
}
