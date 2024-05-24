<template>
  <h1>内存故障预测</h1>
  <br/>
  <img alt="logo" src="./assets/1.webp" :width=200 :height=200>
  <br/><br/>

  <div class='upload-container'>
    <el-upload class="upload" 
      drag
      ref="upload"
      action="http://127.0.0.1:18000/predict"
      :file-list="this.selectedFiles"
      :on-change="handleChange"
      :before-upload="checkfile"
      :on-success="downloadFile"
      :on-error="err"
      response-type="blob"
      :limit="2"
      :auto-upload="false"
      accept=".csv"
      >
      <div class="el-upload__text">
        Drop file(.csv) here or <em>click to upload</em>
      </div>
      
      <template #tip>
        <div class="el-upload__tip">
          files with a size less than 20MB
        </div>
      </template>
    </el-upload>
  </div>
  <el-button class="ml-3" type="success" @click="uploadFile">
        upload to server
      </el-button>

  <br/>
  <!--<h3>{{ flag }}</h3> -->
  <h3 v-show="f">预测结果：</h3>
  <h4 v-show="f">是否发生故障：{{ iswrong }}</h4>
  <h4 v-show="ff">故障发生时间：{{ wrongtime }}</h4>
</template>

<script>
import axios from 'axios';

export default {
  name: 'App',
  data() {
    return {
      selectedFiles: [],
      flag: 0,
      iswrong: "",
      wrongtime: "",
      f: false,
      ff: false,
      g: 0,
    };
  },
  methods: {
    handleFileChange(event) {
      this.selectedFiles = event.target.files;
    },
    handleChange(file, fileList) {
      this.selectedFiles = fileList.slice(-1);
      this.f = false;
      this.ff = false;
    },
    checkfile(file) {
      if (this.selectedFiles.length < 1 ) {
        this.$message.error('未选择文件!');
        return false;
      }
      const isLt2M = this.selectedFiles[0].size / 1024 / 1024 < 20;
      //console.log(this.selectedFiles[0].size)
      if (!isLt2M) {
        this.$message.error('上传文件大小不能超过 20MB!');
      }
      return isLt2M;
    },
    downloadFile(response,file,fileList) {
      console.log(this.selectedFiles)
      //this.selectedFiles = [];
      this.$refs.upload.clearFiles();
      console.log(this.selectedFiles)
      //console.log(response)
      //console.log(response.data);
      const filename="result.csv";
      const url = window.URL.createObjectURL(new Blob([response]));
      const link = document.createElement('a');
      link.href = url;
      link.setAttribute('download', filename);
      document.body.appendChild(link);
      link.click();
      window.URL.revokeObjectURL(url);
    },
    err(err, file, fileList) {
      // console.error('Error uploading files:', error);
      this.$message.error('上传文件失败!');
    },
    uploadFile() {
      this.$refs.upload.submit();
      return;

      console.log("selectedFiles",this.selectedFiles);
      if (!this.checkfile(this.selectedFiles)) {
        return;
      }
      const formData = new FormData();
      /*
      for (const file of this.selectedFiles) {
        //formData.append('file', file);
        this.g = this.g +1;
      }
      console.log(this.g);
      this.g = 0;*/

      //formData.append("file",this.selectedFiles[0])

      console.log("formdata",formData);
      console.log("formdata file",formData.get('file'));

      // http://127.0.0.1:4523/m2/2692086-0-default/169216034
      axios.post('http://127.0.0.1:18000/predict', formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      })
        .then(response => {
          console.log('Files uploaded successfully:', response.data);
          this.selectedFiles = [];
          this.iswrong = response.data.iswrong;
          if (response.data.iswrong == true)
            {
              this.wrongtime = response.data.wrongtime;
              this.ff = true;
            }
          else
            {
              this.wrongtime = "";
              this.ff = false;
            }
          this.flag = 111;
          this.f = true;
        })
        .catch(error => {
          console.error('Error uploading files:', error);
          this.$message.error('上传文件失败!');
          this.flag = 222;
        });
    },
  },
}
</script>

<style>
#app {
  font-family: Avenir, Helvetica, Arial, sans-serif;
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
  text-align: center;
  color: #2c3e50;
  margin-top: 60px;
}

.el-upload .el-upload-dragger {
  width: 500px;
  height: 100px;
}
.upload-container {
  display: flex;
  justify-content: center;
  align-items: center;
  width: 100%;
}
</style>
