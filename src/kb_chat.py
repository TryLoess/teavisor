import os
import time
from typing import Dict, Literal, Tuple

import pandas as pd
import streamlit as st
import streamlit_antd_components as sac
from st_aggrid import AgGrid, JsCode
from st_aggrid.grid_options_builder import GridOptionsBuilder
from streamlit_antd_components.utils import ParseItems

from settings import Settings
from server.knowledge_base.kb_service.base import (
    get_kb_details,
    get_kb_file_details,
)
from server.knowledge_base.utils import LOADER_DICT, get_file_path
from server.utils import get_config_models, get_default_embedding

from webui_pages.utils import *

# SENTENCE_SIZE = 100

cell_renderer = JsCode(
    """function(params) {if(params.value==true){return '?'}else{return '��'}}"""
)


def config_aggrid(
    df: pd.DataFrame,
    columns: Dict[Tuple[str, str], Dict] = {},
    selection_mode: Literal["single", "multiple", "disabled"] = "single",
    use_checkbox: bool = False,
) -> GridOptionsBuilder:
    gb = GridOptionsBuilder.from_dataframe(df)
    gb.configure_column("No", width=40)
    for (col, header), kw in columns.items():
        gb.configure_column(col, header, wrapHeaderText=True, **kw)
    gb.configure_selection(
        selection_mode=selection_mode,
        use_checkbox=use_checkbox,
        pre_selected_rows=st.session_state.get("selected_rows", [0]),
    )
    gb.configure_pagination(
        enabled=True, paginationAutoPageSize=False, paginationPageSize=10
    )
    return gb


def file_exists(kb: str, selected_rows: List) -> Tuple[str, str]:
    """
    check whether a doc file exists in local knowledge base folder.
    return the file's name and path if it exists.
    """
    if selected_rows:
        file_name = selected_rows[0]["file_name"]
        file_path = get_file_path(kb, file_name)
        if os.path.isfile(file_path):
            return file_name, file_path
    return "", ""


def knowledge_base_page(api: ApiRequest, is_lite: bool = None):
    try:
        kb_list = {x["kb_name"]: x for x in get_kb_details()}
    except Exception as e:
        st.error(
            "��ȡ֪ʶ����Ϣ���������Ƿ��Ѱ��� `README.md` �� `4 ֪ʶ���ʼ����Ǩ��` ������ɳ�ʼ����Ǩ�ƣ����Ƿ�Ϊ���ݿ����Ӵ���"
        )
        st.stop()
    kb_names = list(kb_list.keys())

    if (
        "selected_kb_name" in st.session_state
        and st.session_state["selected_kb_name"] in kb_names
    ):
        selected_kb_index = kb_names.index(st.session_state["selected_kb_name"])
    else:
        selected_kb_index = 0

    if "selected_kb_info" not in st.session_state:
        st.session_state["selected_kb_info"] = ""

    def format_selected_kb(kb_name: str) -> str:
        if kb := kb_list.get(kb_name):
            return f"{kb_name} ({kb['vs_type']} @ {kb['embed_model']})"
        else:
            return kb_name

    selected_kb = st.selectbox(
        "��ѡ����½�֪ʶ�⣺",
        kb_names + ["�½�֪ʶ��"],
        format_func=format_selected_kb,
        index=selected_kb_index,
    )

    if selected_kb == "�½�֪ʶ��":
        with st.form("�½�֪ʶ��"):
            kb_name = st.text_input(
                "�½�֪ʶ������",
                placeholder="��֪ʶ�����ƣ���֧����������",
                key="kb_name",
            )
            kb_info = st.text_input(
                "֪ʶ����",
                placeholder="֪ʶ���飬����Agent����",
                key="kb_info",
            )

            col0, _ = st.columns([3, 1])

            vs_types = list(Settings.kb_settings.kbs_config.keys())
            vs_type = col0.selectbox(
                "����������",
                vs_types,
                index=vs_types.index(Settings.kb_settings.DEFAULT_VS_TYPE),
                key="vs_type",
            )

            col1, _ = st.columns([3, 1])
            with col1:
                embed_models = list(get_config_models(model_type="embed"))
                index = 0
                if get_default_embedding() in embed_models:
                    index = embed_models.index(get_default_embedding())
                embed_model = st.selectbox("Embeddingsģ��", embed_models, index)

            submit_create_kb = st.form_submit_button(
                "�½�",
                # disabled=not bool(kb_name),
                use_container_width=True,
            )

        if submit_create_kb:
            if not kb_name or not kb_name.strip():
                st.error(f"֪ʶ�����Ʋ���Ϊ�գ�")
            elif kb_name in kb_list:
                st.error(f"��Ϊ {kb_name} ��֪ʶ���Ѿ����ڣ�")
            elif embed_model is None:
                st.error(f"��ѡ��Embeddingģ�ͣ�")
            else:
                ret = api.create_knowledge_base(
                    knowledge_base_name=kb_name,
                    vector_store_type=vs_type,
                    embed_model=embed_model,
                )
                st.toast(ret.get("msg", " "))
                st.session_state["selected_kb_name"] = kb_name
                st.session_state["selected_kb_info"] = kb_info
                st.rerun()

    elif selected_kb:
        kb = selected_kb
        st.session_state["selected_kb_info"] = kb_list[kb]["kb_info"]
        # �ϴ��ļ�
        files = st.file_uploader(
            "�ϴ�֪ʶ�ļ���",
            [i for ls in LOADER_DICT.values() for i in ls],
            accept_multiple_files=True,
        )
        kb_info = st.text_area(
            "������֪ʶ�����:",
            value=st.session_state["selected_kb_info"],
            max_chars=None,
            key=None,
            help=None,
            on_change=None,
            args=None,
            kwargs=None,
        )

        if kb_info != st.session_state["selected_kb_info"]:
            st.session_state["selected_kb_info"] = kb_info
            api.update_kb_info(kb, kb_info)

        # with st.sidebar:
        with st.expander(
            "�ļ���������",
            expanded=True,
        ):
            cols = st.columns(3)
            chunk_size = cols[0].number_input("�����ı���󳤶ȣ�", 1, 1000, Settings.kb_settings.CHUNK_SIZE)
            chunk_overlap = cols[1].number_input(
                "�����ı��غϳ��ȣ�", 0, chunk_size, Settings.kb_settings.OVERLAP_SIZE
            )
            cols[2].write("")
            cols[2].write("")
            zh_title_enhance = cols[2].checkbox("�������ı����ǿ", Settings.kb_settings.ZH_TITLE_ENHANCE)

        if st.button(
            "����ļ���֪ʶ��",
            # use_container_width=True,
            disabled=len(files) == 0,
        ):
            ret = api.upload_kb_docs(
                files,
                knowledge_base_name=kb,
                override=True,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                zh_title_enhance=zh_title_enhance,
            )
            if msg := check_success_msg(ret):
                st.toast(msg, icon="?")
            elif msg := check_error_msg(ret):
                st.toast(msg, icon="?")

        st.divider()

        # ֪ʶ������
        # st.info("��ѡ���ļ��������ť���в�����")
        doc_details = pd.DataFrame(get_kb_file_details(kb))
        selected_rows = []
        if not len(doc_details):
            st.info(f"֪ʶ�� `{kb}` �������ļ�")
        else:
            st.write(f"֪ʶ�� `{kb}` �������ļ�:")
            st.info("֪ʶ���а���Դ�ļ��������⣬����±���ѡ���ļ������")
            doc_details.drop(columns=["kb_name"], inplace=True)
            doc_details = doc_details[
                [
                    "No",
                    "file_name",
                    "document_loader",
                    "text_splitter",
                    "docs_count",
                    "in_folder",
                    "in_db",
                ]
            ]
            doc_details["in_folder"] = (
                doc_details["in_folder"].replace(True, "?").replace(False, "��")
            )
            doc_details["in_db"] = (
                doc_details["in_db"].replace(True, "?").replace(False, "��")
            )
            gb = config_aggrid(
                doc_details,
                {
                    ("No", "���"): {},
                    ("file_name", "�ĵ�����"): {},
                    # ("file_ext", "�ĵ�����"): {},
                    # ("file_version", "�ĵ��汾"): {},
                    ("document_loader", "�ĵ�������"): {},
                    ("docs_count", "�ĵ�����"): {},
                    ("text_splitter", "�ִ���"): {},
                    # ("create_time", "����ʱ��"): {},
                    ("in_folder", "Դ�ļ�"): {},
                    ("in_db", "������"): {},
                },
                "multiple",
            )

            doc_grid = AgGrid(
                doc_details,
                gb.build(),
                columns_auto_size_mode="FIT_CONTENTS",
                theme="alpine",
                custom_css={
                    "#gridToolBar": {"display": "none"},
                },
                allow_unsafe_jscode=True,
                enable_enterprise_modules=False,
            )

            selected_rows = doc_grid.get("selected_rows")
            if selected_rows is None:
                selected_rows = []
            else:
                selected_rows = selected_rows.to_dict("records")
            cols = st.columns(4)
            file_name, file_path = file_exists(kb, selected_rows)
            if file_path:
                with open(file_path, "rb") as fp:
                    cols[0].download_button(
                        "����ѡ���ĵ�",
                        fp,
                        file_name=file_name,
                        use_container_width=True,
                    )
            else:
                cols[0].download_button(
                    "����ѡ���ĵ�",
                    "",
                    disabled=True,
                    use_container_width=True,
                )

            st.write()
            # ���ļ��ִʲ����ص���������
            if cols[1].button(
                "���������������"
                if selected_rows and (pd.DataFrame(selected_rows)["in_db"]).any()
                else "�����������",
                disabled=not file_exists(kb, selected_rows)[0],
                use_container_width=True,
            ):
                file_names = [row["file_name"] for row in selected_rows]
                api.update_kb_docs(
                    kb,
                    file_names=file_names,
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap,
                    zh_title_enhance=zh_title_enhance,
                )
                st.rerun()

            # ���ļ�����������ɾ��������ɾ���ļ�����
            if cols[2].button(
                "��������ɾ��",
                disabled=not (selected_rows and selected_rows[0]["in_db"]),
                use_container_width=True,
            ):
                file_names = [row["file_name"] for row in selected_rows]
                api.delete_kb_docs(kb, file_names=file_names)
                st.rerun()

            if cols[3].button(
                "��֪ʶ����ɾ��",
                type="primary",
                use_container_width=True,
            ):
                file_names = [row["file_name"] for row in selected_rows]
                api.delete_kb_docs(kb, file_names=file_names, delete_content=True)
                st.rerun()

        st.divider()

        cols = st.columns(3)

        if cols[0].button(
            "����Դ�ļ��ؽ�������",
            help="�����ϴ��ļ���ͨ��������ʽ���ĵ���������Ӧ֪ʶ��contentĿ¼�£��������ť�����ؽ�֪ʶ�⡣",
            use_container_width=True,
            type="primary",
        ):
            with st.spinner("�������ع��У������ĵȴ�����ˢ�»�ر�ҳ�档"):
                empty = st.empty()
                empty.progress(0.0, "")
                for d in api.recreate_vector_store(
                    kb,
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap,
                    zh_title_enhance=zh_title_enhance,
                ):
                    if msg := check_error_msg(d):
                        st.toast(msg)
                    else:
                        empty.progress(d["finished"] / d["total"], d["msg"])
                st.rerun()

        if cols[2].button(
            "ɾ��֪ʶ��",
            use_container_width=True,
        ):
            ret = api.delete_knowledge_base(kb)
            st.toast(ret.get("msg", " "))
            time.sleep(1)
            st.rerun()

        with st.sidebar:
            keyword = st.text_input("��ѯ�ؼ���")
            top_k = st.slider("ƥ������", 1, 100, 3)

        st.write("�ļ����ĵ��б�˫�������޸ģ���ɾ�������� Y ��ɾ����Ӧ�С�")
        docs = []
        df = pd.DataFrame([], columns=["seq", "id", "content", "source"])
        if selected_rows:
            file_name = selected_rows[0]["file_name"]
            docs = api.search_kb_docs(
                knowledge_base_name=selected_kb, file_name=file_name
            )

            data = [
                {
                    "seq": i + 1,
                    "id": x["id"],
                    "page_content": x["page_content"],
                    "source": x["metadata"].get("source"),
                    "type": x["type"],
                    "metadata": json.dumps(x["metadata"], ensure_ascii=False),
                    "to_del": "",
                }
                for i, x in enumerate(docs)
            ]
            df = pd.DataFrame(data)

            gb = GridOptionsBuilder.from_dataframe(df)
            gb.configure_columns(["id", "source", "type", "metadata"], hide=True)
            gb.configure_column("seq", "No.", width=50)
            gb.configure_column(
                "page_content",
                "����",
                editable=True,
                autoHeight=True,
                wrapText=True,
                flex=1,
                cellEditor="agLargeTextCellEditor",
                cellEditorPopup=True,
            )
            gb.configure_column(
                "to_del",
                "ɾ��",
                editable=True,
                width=50,
                wrapHeaderText=True,
                cellEditor="agCheckboxCellEditor",
                cellRender="agCheckboxCellRenderer",
            )
            # ���÷�ҳ
            gb.configure_pagination(
                enabled=True, paginationAutoPageSize=False, paginationPageSize=10
            )
            gb.configure_selection()
            edit_docs = AgGrid(df, gb.build(), fit_columns_on_grid_load=True)

            if st.button("�������"):
                origin_docs = {
                    x["id"]: {
                        "page_content": x["page_content"],
                        "type": x["type"],
                        "metadata": x["metadata"],
                    }
                    for x in docs
                }
                changed_docs = []
                for index, row in edit_docs.data.iterrows():
                    origin_doc = origin_docs[row["id"]]
                    if row["page_content"] != origin_doc["page_content"]:
                        if row["to_del"] not in ["Y", "y", 1]:
                            changed_docs.append(
                                {
                                    "page_content": row["page_content"],
                                    "type": row["type"],
                                    "metadata": json.loads(row["metadata"]),
                                }
                            )

                if changed_docs:
                    if api.update_kb_docs(
                        knowledge_base_name=selected_kb,
                        file_names=[file_name],
                        docs={file_name: changed_docs},
                    ):
                        st.toast("�����ĵ��ɹ�")
                    else:
                        st.toast("�����ĵ�ʧ��")
