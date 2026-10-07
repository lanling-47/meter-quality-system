import { Presentation, PresentationFile } from '@oai/artifact-tool';
import fs from 'node:fs/promises';

const OUT = 'D:/biye/meter-quality-system/智能电表质量管控系统_企业使用与操作指南.pptx';
const IMG = 'D:/biye/meter-quality-system/ppt_build/screens';
const W = 1280, H = 720;
const C = { navy:'#1A3A5C', blue:'#2E6F9E', yellow:'#F0A500', red:'#E55245', green:'#1FA968', ink:'#17324D', gray:'#5C6B78', light:'#F4F7FA', line:'#D9E1E8', white:'#FFFFFF' };
const p = Presentation.create({slideSize:{width:W,height:H}});

async function bytes(path){ const b=await fs.readFile(path); return b.buffer.slice(b.byteOffset,b.byteOffset+b.byteLength); }
function box(s,x,y,w,h,fill=C.white,line=C.line,geom='roundRect'){ return s.shapes.add({geometry:geom,position:{left:x,top:y,width:w,height:h},fill,line:{style:'solid',fill:line,width:line==='none'?0:1},borderRadius:geom==='roundRect'?'rounded-lg':undefined}); }
function txt(s,text,x,y,w,h,size=20,color=C.ink,bold=false,align='left'){ const t=s.shapes.add({geometry:'textbox',position:{left:x,top:y,width:w,height:h},fill:'none',line:{style:'solid',fill:'none',width:0}}); t.text=text; t.text.style={fontSize:size,color,bold,alignment:align,fontFamily:'Arial'}; return t; }
function title(s,kicker,heading,sub=''){ txt(s,kicker.toUpperCase(),72,42,420,24,14,C.blue,true); txt(s,heading,72,74,1120,58,36,C.navy,true); if(sub) txt(s,sub,74,138,1110,34,18,C.gray,false); }
function footer(s,n){ txt(s,'智能电表全周期质量管控系统  |  企业使用与操作指南',72,682,700,18,12,C.gray); txt(s,String(n).padStart(2,'0'),1148,680,58,18,13,C.gray,true,'right'); }
async function image(s,file,x,y,w,h){ s.images.add({blob:await bytes(`${IMG}/${file}`),contentType:'image/png',fit:'cover',position:{left:x,top:y,width:w,height:h},geometry:'roundRect',borderRadius:'rounded-lg'}); }
function pill(s,label,x,y,w,fill,color=C.white){ box(s,x,y,w,30,fill,fill); txt(s,label,x,y+5,w,20,14,color,true,'center'); }

// 1 cover
{ const s=p.slides.add(); s.background.fill=C.navy; txt(s,'企业现场操作与应用指南',82,72,420,28,18,'#A9C8E1',true); txt(s,'智能电表\n全周期质量管控系统',82,126,620,150,52,C.white,true); txt(s,'把分散的质量数据，变成看得见、查得清、能处置的现场动作。',86,306,570,56,23,'#DCEAF4'); pill(s,'LOCAL DEPLOYMENT',86,405,190,C.yellow,C.navy); await image(s,'04-dashboard.png',720,70,470,520); txt(s,'质量看板 · 风险预测 · 根因追溯 · 现场处置单',86,620,640,26,17,'#BFD8EA'); }

// 2 why
{ const s=p.slides.add(); s.background.fill=C.white; title(s,'01 先看价值','让质量管理从“事后追查”变成“现场先处置”','企业一线最需要的，不是更多报表，而是下一步该做什么。');
const items=[['数据散落','Excel、老化、出厂记录分开存放','同一块表难以串起来'],['异常难发现','人工筛表、靠经验判断','高风险可能被遗漏'],['交接不统一','复测、隔离、通知各写各的','班组之间容易断档'],['决策缺证据','只知道“有问题”','说不清问题在哪里']];
items.forEach((it,i)=>{const x=78+(i%2)*560,y=220+Math.floor(i/2)*170; box(s,x,y,510,130,C.light,C.line); txt(s,it[0],x+24,y+20,180,28,24,C.navy,true); txt(s,it[1],x+24,y+58,450,24,18,C.ink); txt(s,it[2],x+24,y+88,450,22,16,C.gray);}); footer(s,2); }

//3 flow
{ const s=p.slides.add(); s.background.fill=C.white; title(s,'02 一条主线','操作员只需记住这 7 步','从数据准备到处置闭环，每一步都有明确入口。'); const steps=[['1','准备数据'],['2','导入/监听'],['3','看质量看板'],['4','风险预测'],['5','根因追溯'],['6','复测处置'],['7','导出处置单']]; steps.forEach((st,i)=>{const x=70+i*170; if(i<6){const ln=s.shapes.add({geometry:'line',position:{left:x+104,top:300,width:65,height:1},line:{style:'solid',fill:C.line,width:3},fill:'none'});} box(s,x,255,110,110,i===6?C.yellow:C.navy,i===6?C.yellow:C.navy); txt(s,st[0],x,268,110,35,28,C.white,true,'center'); txt(s,st[1],x-25,390,160,30,18,C.navy,true,'center'); }); box(s,110,500,1060,74,'#FFF7E5','#F4D38A'); txt(s,'现场原则：高风险先隔离复测；模型结果用于辅助排查，不替代检验放行。',142,522,1000,30,21,C.navy,true,'center'); footer(s,3); }

//4 login setup
{ const s=p.slides.add(); s.background.fill=C.white; title(s,'03 第一步','先确认系统和数据是否准备好','登录后先完成部署检查，再进入业务页面。'); await image(s,'01-login.png',72,205,520,350); await image(s,'02-setup.png',650,205,520,350); pill(s,'A',92,180,34,C.yellow,C.navy); pill(s,'B',670,180,34,C.yellow,C.navy); txt(s,'登录：账号 admin，密码 admin123',72,575,520,28,18,C.ink,true); txt(s,'部署向导：检查数据目录、样本数和监听目录',650,575,520,28,18,C.ink,true); footer(s,4); }

//5 import
{ const s=p.slides.add(); s.background.fill=C.white; title(s,'04 第二步','把 Excel 放进系统，先看导入结果','导入完成后，核对数量和错误，再开始分析。'); await image(s,'03-import.png',72,200,720,400); box(s,840,205,330,390,C.light,C.line); txt(s,'操作顺序',875,235,250,30,22,C.navy,true); ['选择数据类型','上传 Excel 或放入监听目录','查看成功 / 失败数量','核对条码、表号、批次'].forEach((v,i)=>{pill(s,String(i+1),875,292+i*62,34,C.yellow,C.navy); txt(s,v,925,298+i*62,210,24,17,C.ink,i===3);}); txt(s,'关键检查：barcode ↔ meter_no 必须能对应。',875,548,250,42,16,C.red,true); footer(s,5); }

//6 dashboard
{ const s=p.slides.add(); s.background.fill=C.white; title(s,'05 第三步','先看高风险，再安排复测','看板把全流程状态集中到一个页面。'); await image(s,'04-dashboard.png',72,190,820,455); box(s,930,205,260,385,C.navy,C.navy); txt(s,'现场先看 4 个数',956,235,210,30,21,C.white,true); [['31','老化试验总数'],['11','高风险电表'],['20','质量正常'],['180','老化试记录']].forEach((a,i)=>{txt(s,a[0],956,292+i*64,70,30,26,i===1? '#FF9C91':'#FFFFFF',true); txt(s,a[1],1030,298+i*64,130,22,16,'#D5E6F2');}); txt(s,'黄色按钮 = 现场动作\n红色 = 优先处置\n绿色 = 当前正常',956,515,200,60,16,'#D5E6F2'); footer(s,6); }

//7 predict
{ const s=p.slides.add(); s.background.fill=C.white; title(s,'06 第四步','确认模型已激活，再批量预测','模型是“辅助预警器”，不是自动放行器。'); await image(s,'05-predict.png',72,200,735,410); box(s,845,205,345,410,'#FFF7E5','#F4D38A'); txt(s,'操作要点',875,235,240,28,23,C.navy,true); ['看激活模型状态','点击批量重新预测','按高 / 中 / 低风险分流','高风险：隔离 + 复测'].forEach((v,i)=>{pill(s,String(i+1),875,292+i*58,34,C.yellow,C.navy); txt(s,v,925,298+i*58,230,24,17,C.ink,i===3);}); txt(s,'演示模型指标：AUC 0.995，F1 0.957\n仅代表当前样例训练集表现。',875,525,270,55,15,C.red,true); footer(s,7); }

//8 root/degradation
{ const s=p.slides.add(); s.background.fill=C.white; title(s,'07 第五步','用根因和趋势，告诉班组“为什么”','预测给出风险，分析页面提供复测线索。'); await image(s,'06-rootcause.png',72,205,520,350); await image(s,'07-degradation.png',650,205,520,350); txt(s,'根因追溯：定位异常指标与关联工序',72,580,520,28,18,C.navy,true); txt(s,'退化分析：查看误差、时钟等指标趋势',650,580,520,28,18,C.navy,true); footer(s,8); }

//9 report/help
{ const s=p.slides.add(); s.background.fill=C.white; title(s,'08 第六步','报告直接告诉你“现在怎么做”','一键导出现场处置单，班组拿到就能执行。'); await image(s,'08-help.png',72,195,470,410); box(s,590,205,590,360,C.light,C.line); txt(s,'智能电表现场质量处置单',625,235,500,34,27,C.navy,true); [['电表编号','#196 / #201'],['发现的问题','末期误差超限 / 高风险'],['现在怎么做','立即复测，必要时隔离'],['现场勾选','复测  隔离  通知  记录']].forEach((a,i)=>{txt(s,a[0],625,300+i*55,130,24,17,C.gray,true); txt(s,a[1],785,300+i*55,330,24,17,i===2?C.red:C.ink,i===2);}); pill(s,'导出现场处置单',625,520,190,C.yellow,C.navy); txt(s,'帮助侧栏：在任何页面点“帮助”，按当前页面给出操作提示。',625,580,510,25,16,C.gray); footer(s,9); }

//10 boundaries
{ const s=p.slides.add(); s.background.fill=C.navy; txt(s,'09 试点落地建议',80,56,400,26,16,'#A9C8E1',true); txt(s,'先把它用起来，再把它做深',80,96,900,60,40,C.white,true); txt(s,'建议以一个产线、一个批次、一个班组开始试点。',82,165,780,30,20,'#DCEAF4'); const left=[['适合现在做','辅助预警、复测安排、数据集中管理','现场操作指引与处置单'],['上线前必须确认','真实生产数据质量','barcode ↔ meter_no 对应关系','检验规则与模型阈值'],['后续可增强','角色权限与审计','处置结果回写','备份、报表与多产线对比']]; left.forEach((a,i)=>{const y=250+i*120; box(s,82,y,500,94,i===0?'#244C70':'#203F5C',i===0?C.yellow:'#39698D'); txt(s,a[0],108,y+16,170,25,21,C.white,true); txt(s,a[1],300,y+14,245,23,16,'#E1EEF6'); if(a[2]) txt(s,a[2],300,y+45,245,23,16,'#E1EEF6');}); box(s,660,250,500,300,C.white,C.white); txt(s,'给企业的承诺',700,285,380,32,24,C.navy,true); txt(s,'看得懂\n查得清\n能处置\n可追溯',700,340,260,170,30,C.navy,true); txt(s,'模型结果辅助判断，最终放行仍由检验制度负责。',700,560,400,24,16,C.red,true); txt(s,'智能电表全周期质量管控系统  |  企业使用与操作指南',82,670,700,18,12,'#BFD8EA'); txt(s,'10',1148,668,58,18,13,'#BFD8EA',true,'right'); }

const pptx = await PresentationFile.exportPptx(p); await pptx.save(OUT);
await fs.mkdir('D:/biye/meter-quality-system/ppt_build/rendered', {recursive:true});
for (const [i,sl] of p.slides.items.entries()) { const b=await p.export({slide:sl,format:'png',scale:1}); await fs.writeFile(`D:/biye/meter-quality-system/ppt_build/rendered/slide-${i+1}.png`, Buffer.from(await b.arrayBuffer())); }
const montage=await p.export({format:'webp',montage:true,scale:1}); await fs.writeFile('D:/biye/meter-quality-system/ppt_build/rendered/montage.webp', Buffer.from(await montage.arrayBuffer()));
console.log(OUT);
