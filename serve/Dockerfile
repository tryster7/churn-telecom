# See the License for the specific language governing permissions and
# limitations under the License.

FROM bitnami/kubectl

ADD deploy.sh tfserve.yaml /tmp/
USER root
RUN /bin/bash -c 'chmod +x /tmp/deploy.sh'

ENTRYPOINT ["/tmp/deploy.sh"]
