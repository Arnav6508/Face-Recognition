<a id="readme-top"></a>

<br />
<div align="center">
  <h3 align="center">Attendance web app</h3>

  <p align="center">
    Face recognition attendance system using YOLO
    <br />
    <a href="https://github.com/Arnav6508/Attendance-web-app"><strong>Explore the docs »</strong></a>
    <br />
  </p>
</div>


<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#installation">Installation</a></li>
      </ul>
    </li>
    <li><a href="#contributing">Contributing</a></li>
    <li><a href="#contact">Contact</a></li>
  </ol>
</details>



<!-- ABOUT THE PROJECT -->
## About The Project

This project is a Face Recognition-Based Attendance System, designed as a desktop application with a Tkinter-based frontend and a SQLite database for efficient data management. The system leverages YOLO for real-time face detection and a pretrained InceptionResNetV1 model for face recognition using one-shot learning.



<!-- GETTING STARTED -->
## Getting Started

<h4>1. Attendance app:</h4>
<ul>
    <li><b>File name:</b> main.py</li>
    <li><b>Pre-requisites:</b>
        <ul>
            <li>face detection model (model)</li>
            <li>face recognition model (base_model)</li>
            <li>SQL database (embeddings_db_path)</li>
        </ul>
    </li>
</ul>

<h4>2. Addressing the Pre-requisites:</h4>
<ul>
    <li><b>Face detection:</b>
        <ul>
            <li>Used YOLO v8 over 100 epochs</li>
            <li>Files: scripts/train.py and config.yaml</li>
            <li>model weights path: best.pt</li>
        </ul>
    </li>
    <li><b>Face recognition:</b>
        <ul>
            <li>Used InceptionResnetV1 pretrained on vggface2 dataset (Nothing to do here)</li>
        </ul>
    </li>
    <li><b>Database:</b>
        <ul>
            <li>Utilised SQL database through sqlite3</li>
            <li>table1 = Embeddings(<br>
                &emsp;name TEXT,<br>
                &emsp;embedding BLOB<br>
            )</li>
            <li>table2 = Attendace(<br>
                &emsp;person_id INTEGER,<br>
                &emsp;name TEXT NOT NULL,<br>
                &emsp;date DATE NOT NULL,<br>
                &emsp;time TIME NOT NULL,<br>
                &emsp;FOREIGN KEY (person_id) REFERENCES Embeddings (id)<br>
            )</li>
            <li>default file name: test.db</li>
        </ul>
    </li>
</ul>

<h4>3. Inference:</h4>
Inside scripts/test.py, 3 inference functions are provided:<br>
- <b>test_from_webcam:</b> Real-time face recognition<br>
- <b>test_from_path:</b> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Provides name of person from image path<br>
- <b>test_from_image:</b> &nbsp;&nbsp;&nbsp;&nbsp;Provides name of person from image (For internal use)<br>
<br>



### Installation

_Below is an example of how you can instruct your audience on installing and setting up your app._

1. Clone the repo
   ```sh
   git clone https://github.com/Arnav6508/Attendance-web-app
   ```

2. Change git remote url to avoid accidental pushes to base project
   ```sh
   git remote set-url origin github_username/repo_name
   git remote -v # confirm the changes
   ```


<!-- CONTRIBUTING -->
## Contributing

Contributions are what make the open source community such an amazing place to learn, inspire, and create. Any contributions you make are **greatly appreciated**.

If you have a suggestion that would make this better, please fork the repo and create a pull request. You can also simply open an issue with the tag "enhancement".
Don't forget to give the project a star! Thanks again!

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request


<!-- CONTACT -->
## Contact

Email -  arnavgupta6508@gmail.com


<p align="right">(<a href="#readme-top">back to top</a>)</p>

