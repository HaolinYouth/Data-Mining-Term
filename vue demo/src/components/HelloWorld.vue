<template>
  <div class="hello">
    <h2>{{ msg }}</h2>
    <h1>{{ message }}</h1>
    <button @click = "count+=1">click counter {{ count }}</button><br/><br/>
    <button @click = "test">{{ tt }}</button><br/><br/>

    <input type = "text" v-model.lazy="username">
    <p>{{ username }}</p>
  </div>
</template>

<script>

export default {
  name: 'HelloWorld',
  props:{
    msg:{
      type:String,
      default:""
    }
  },
  data(){
    return {
      message:"abcdefg",
      count:0,
      tt: "asd",
      username: "",
      get_username: ""
    }
  },
  methods: {
    test(){
      this.message = this.message.split('').reverse().join('');
      this.tt = this.tt.split('').reverse().join('');
      //alert(this.msg);
    }
  }
}
</script>
